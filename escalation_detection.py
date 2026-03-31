"""Escalation Detection Processor for Pipecat voice pipelines.

Monitors every user transcription turn and detects when the conversation
should be handed off to a human agent.  Emits an EscalationFrame (and
fires the "on_escalation_triggered" event) exactly once per call.

Detection triggers
──────────────────
1. Explicit request   — Caller directly asks for a human / manager /
                        supervisor / transfer (English, Hindi Devanagari,
                        Romanized Hindi).  Fires immediately.

2. Frustration        — Accumulates a score from negative-sentiment
                        keywords across turns.  Fires when score ≥ 2.0
                        (roughly 2–3 strong signals or 4–5 weak ones).

3. Loop               — Last 3 consecutive user messages each have
                        ≥ 4 words AND pairwise Jaccard word-overlap ≥ 0.55.
                        This catches a caller re-stating the same question
                        because the agent keeps failing to resolve it.

4. Out-of-domain      — Caller explicitly asks about topics clearly outside
                        Kotak Securities scope (medical, travel, food, etc.)
                        while also expressing that the agent cannot help.
                        Fires when 2+ out-of-domain turns are detected.

All frames continue to pass through unchanged; this processor is purely
observational except for emitting the EscalationFrame.
"""

import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


EscalationReason = Literal["explicit_request", "frustration", "loop", "out_of_domain"]


@dataclass
class EscalationFrame(Frame):
    """Emitted when the conversation should be handed off to a human agent."""

    reason: EscalationReason            # why escalation was triggered
    trigger_text: str                   # the utterance that tipped the scale
    turn_number: int                    # conversation turn at which it fired
    frustration_score: float            # accumulated frustration score
    transcript: list[str]               # all user turns recorded so far


# ── Normalization ──────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ── 1. Explicit escalation signals ────────────────────────────────────────

# English regex patterns
_EXPLICIT_EN: list[str] = [
    r"\bhuman\b",
    r"\breal\s+(?:person|agent|human)\b",
    r"\blive\s+(?:agent|person)\b",
    r"\bmanager\b",
    r"\bsupervisor\b",
    r"\brepresentative\b",
    r"\bspeak\s+(?:to|with)\s+(?:someone|a\s+person|an?\s+agent|a\s+human)\b",
    r"\btalk\s+(?:to|with)\s+(?:someone|a\s+person|an?\s+agent|a\s+human)\b",
    r"\btransfer\s+(?:me|my\s+call|this\s+call)\b",
    r"\bconnect\s+me\b",
    r"\bcustomer\s+(?:care|service|support)\b",
    r"\bescalat\b",
    r"\bcall\s+(?:center|centre)\b",
]

# Hindi Devanagari strings (substring match after NFC-normalize)
_EXPLICIT_DEVA: list[str] = [
    "मैनेजर", "सुपरवाइजर", "ट्रांसफर",
    "किसी और से बात", "इंसान से बात", "असली इंसान",
    "कस्टमर केयर", "कस्टमर सर्विस",
    "किसी इंसान", "किसी व्यक्ति",
]

# Romanized Hindi regex patterns
_EXPLICIT_ROMAN: list[str] = [
    r"\bmanager\b", r"\bsupervisor\b",
    r"\binsaan\s+se\b", r"\bkisi\s+aur\s+se\b",
    r"\btransfer\s+kar\b", r"\bconnect\s+kar\b",
    r"\bcustomer\s+care\b", r"\bcustomer\s+service\b",
    r"\bkisi\s+(?:insaan|vyakti|aadmi)\b",
]


def _is_explicit_escalation(text: str) -> bool:
    """Return True if the utterance contains an explicit escalation request."""
    text_norm = _norm(text)
    t_lower = text_norm.lower()

    for pat in _EXPLICIT_EN:
        if re.search(pat, t_lower):
            return True
    for marker in _EXPLICIT_DEVA:
        if _norm(marker) in text_norm:
            return True
    for pat in _EXPLICIT_ROMAN:
        if re.search(pat, t_lower):
            return True
    return False


# ── 2. Frustration signals ─────────────────────────────────────────────────

# (pattern, weight) — weight accumulates toward threshold of 2.0
_FRUSTRATION_EN: list[tuple[str, float]] = [
    (r"\bstop\s+calling\b", 1.2),
    (r"\bleave\s+me\s+alone\b", 1.2),
    (r"\bwaste\s+of\s+(?:my\s+)?time\b", 1.0),
    (r"\bnot\s+(?:interested|helpful|working)\b", 0.7),
    (r"\buseless\b", 0.8),
    (r"\bfed\s+up\b", 0.9),
    (r"\bfrustrat\b", 0.8),
    (r"\bangry\b", 0.9),
    (r"\bannoy\b", 0.7),
    (r"\bdon'?t\s+(?:call|bother|want)\b", 0.6),
    (r"\bstop\s+(?:this|it|now)\b", 0.6),
    (r"\bno\s+(?:more|thanks|way)\b", 0.4),
    (r"\bthis\s+is\s+(?:ridiculous|absurd|awful)\b", 1.0),
]

_FRUSTRATION_DEVA: list[tuple[str, float]] = [
    ("नहीं चाहिए", 0.7),
    ("बंद करो", 0.8),
    ("बेकार", 0.8),
    ("परेशान", 0.7),
    ("बकवास", 0.9),
    ("समय बर्बाद", 1.0),
    ("गुस्सा", 0.9),
    ("नाराज", 0.8),
    ("छोड़ो", 0.6),
]

_FRUSTRATION_ROMAN: list[tuple[str, float]] = [
    (r"\bnahi\s+chahiye\b", 0.7),
    (r"\bband\s+karo\b", 0.8),
    (r"\bbekar\b", 0.8),
    (r"\bbakwaas\b", 0.9),
    (r"\bpareshan\b", 0.7),
    (r"\bgussa\b", 0.9),
    (r"\btime\s+waste\b", 1.0),
    (r"\bchhodo\b", 0.5),
]


def _frustration_score(text: str) -> float:
    """Return a frustration weight for one utterance (0.0 = neutral)."""
    text_norm = _norm(text)
    t_lower = text_norm.lower()
    score = 0.0

    for pat, weight in _FRUSTRATION_EN:
        if re.search(pat, t_lower):
            score += weight
    for marker, weight in _FRUSTRATION_DEVA:
        if _norm(marker) in text_norm:
            score += weight
    for pat, weight in _FRUSTRATION_ROMAN:
        if re.search(pat, t_lower):
            score += weight

    return score


# ── 3. Loop detection ─────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    """Word-set Jaccard similarity between two strings."""
    words_a = set(re.findall(r"\b\w+\b", a.lower()))
    words_b = set(re.findall(r"\b\w+\b", b.lower()))
    if not words_a and not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


_LOOP_WINDOW = 3        # number of consecutive turns to compare
_LOOP_MIN_WORDS = 4     # ignore short turns (e.g. "yes", "ok")
_LOOP_SIMILARITY = 0.55 # Jaccard threshold


def _is_loop(recent_turns: list[str]) -> bool:
    """Return True if the last _LOOP_WINDOW turns are all similar multi-word messages."""
    if len(recent_turns) < _LOOP_WINDOW:
        return False

    window = recent_turns[-_LOOP_WINDOW:]
    # Each turn must have enough words to be meaningful
    if any(len(re.findall(r"\b\w+\b", t)) < _LOOP_MIN_WORDS for t in window):
        return False

    # All pairs in the window must be similar
    for i in range(len(window)):
        for j in range(i + 1, len(window)):
            if _jaccard(window[i], window[j]) < _LOOP_SIMILARITY:
                return False
    return True


# ── 4. Out-of-domain signals ───────────────────────────────────────────────

_OOD_EN: list[str] = [
    r"\b(?:book|flight|hotel|restaurant|food|pizza|taxi|cab|uber)\b",
    r"\b(?:doctor|hospital|medicine|prescription|symptom)\b",
    r"\b(?:cricket|match|score|football|movie|film)\b",
    r"\b(?:weather|forecast|temperature)\b",
    r"\bcan'?t\s+(?:help|answer|assist)\b",
    r"\bdon'?t\s+(?:know|understand)\s+(?:about|this)\b",
]

_OOD_THRESHOLD = 2  # fires after 2 distinct OOD signals


def _has_ood_signal(text: str) -> bool:
    t_lower = _norm(text).lower()
    return any(re.search(pat, t_lower) for pat in _OOD_EN)


# ── Processor ─────────────────────────────────────────────────────────────

_FRUSTRATION_THRESHOLD = 2.0


class EscalationDetectionProcessor(FrameProcessor):
    """Detects when a conversation should be escalated to a human agent.

    Place AFTER the STT service and BEFORE the user aggregator so the
    detector sees every user turn before it reaches the LLM.

    Emits an :class:`EscalationFrame` exactly once per call, then stops
    monitoring (subsequent turns pass through without analysis).

    Example::

        pipeline = Pipeline([
            transport.input(),
            stt,
            language_detector,
            escalation_detector,   # ← insert here
            gender_detector,
            user_aggregator,
            ...
        ])

        @escalation_detector.event_handler("on_escalation_triggered")
        async def on_escalation(processor, frame: EscalationFrame):
            logger.warning(f"Escalation: {frame.reason}")
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._turn: int = 0
        self._transcript: list[str] = []
        self._frustration: float = 0.0
        self._ood_count: int = 0
        self._recent: deque[str] = deque(maxlen=_LOOP_WINDOW)
        self._escalated: bool = False
        self._register_event_handler("on_escalation_triggered")

    # ── Public state ──────────────────────────────────────────────────────

    @property
    def is_escalated(self) -> bool:
        return self._escalated

    @property
    def frustration_score(self) -> float:
        return self._frustration

    def reset(self):
        """Reset state for a new call."""
        self._turn = 0
        self._transcript.clear()
        self._frustration = 0.0
        self._ood_count = 0
        self._recent.clear()
        self._escalated = False

    # ── Frame processing ──────────────────────────────────────────────────

    async def _trigger(
        self, reason: EscalationReason, trigger_text: str, direction: FrameDirection
    ) -> None:
        if self._escalated:
            return
        self._escalated = True

        frame = EscalationFrame(
            reason=reason,
            trigger_text=trigger_text,
            turn_number=self._turn,
            frustration_score=round(self._frustration, 2),
            transcript=list(self._transcript),
        )

        logger.warning(
            f"🚨 EscalationDetection | reason={reason} | turn={self._turn} | "
            f"frustration={self._frustration:.2f} | text='{trigger_text}'"
        )

        await self._call_event_handler("on_escalation_triggered", frame)
        await self.push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, TranscriptionFrame) or not frame.text.strip():
            await self.push_frame(frame, direction)
            return

        # Already escalated — pass through without analysis
        if self._escalated:
            await self.push_frame(frame, direction)
            return

        text = frame.text.strip()
        self._turn += 1
        self._transcript.append(text)
        self._recent.append(text)

        try:
            # ── Trigger 1: explicit escalation request ──
            if _is_explicit_escalation(text):
                logger.debug(f"EscalationDetection | explicit_request | turn={self._turn}")
                await self._trigger("explicit_request", text, direction)
                await self.push_frame(frame, direction)
                return

            # ── Trigger 2: frustration accumulation ──
            turn_frustration = _frustration_score(text)
            if turn_frustration > 0:
                self._frustration += turn_frustration
                logger.debug(
                    f"EscalationDetection | frustration +{turn_frustration:.2f} "
                    f"→ total {self._frustration:.2f} | turn={self._turn}"
                )
            if self._frustration >= _FRUSTRATION_THRESHOLD:
                await self._trigger("frustration", text, direction)
                await self.push_frame(frame, direction)
                return

            # ── Trigger 3: loop detection ──
            if _is_loop(list(self._recent)):
                logger.debug(f"EscalationDetection | loop_detected | turn={self._turn}")
                await self._trigger("loop", text, direction)
                await self.push_frame(frame, direction)
                return

            # ── Trigger 4: out-of-domain accumulation ──
            if _has_ood_signal(text):
                self._ood_count += 1
                logger.debug(
                    f"EscalationDetection | ood_signal | count={self._ood_count} | turn={self._turn}"
                )
                if self._ood_count >= _OOD_THRESHOLD:
                    await self._trigger("out_of_domain", text, direction)
                    await self.push_frame(frame, direction)
                    return

        except Exception as exc:
            logger.warning(f"EscalationDetection | analysis error: {exc}")

        await self.push_frame(frame, direction)
