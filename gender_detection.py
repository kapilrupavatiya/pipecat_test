"""Gender Detection Processor for Pipecat voice pipelines.

Analyzes each user transcription turn and maintains a running gender
estimate with confidence score. Uses rule-based signals:
  - Hindi grammatical gender markers (verb endings: raha/rahi, tha/thi)
  - Hindi address terms (bhai/sir vs didi/madam/behen)
  - English pronouns (he/him/his vs she/her/hers)
  - Common Indian first names

Emits a GenderDetectionFrame after each turn with the current estimate.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import AsyncGenerator, Literal, Optional

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


GenderLabel = Literal["male", "female", "unknown"]


@dataclass
class GenderDetectionFrame(Frame):
    """Carries the current gender estimate after a transcription turn."""

    gender: GenderLabel
    confidence: float          # 0.0 – 1.0
    turn_text: str             # the transcript that updated the estimate
    cumulative_score: float    # raw running score (positive = male, negative = female)


# ── Signal tables ──────────────────────────────────────────────────────────

# Romanized Hindi masculine verb/adjective endings
_HINDI_MALE_ROMAN = [
    r"\bkr?\s*raha\b", r"\bkar\s*raha\b",
    r"\bho\s*raha\b", r"\btha\b",
    r"\bgaya\b", r"\baaya\b", r"\bbola\b", r"\bsota\b",
    r"\bkhaya\b", r"\bpiya\b", r"\bbaitha\b",
    r"\bpita\b", r"\bbeta\b", r"\bbhai\b", r"\buncle\b",
]

# Romanized Hindi feminine verb/adjective endings
_HINDI_FEMALE_ROMAN = [
    r"\bkr?\s*rahi\b", r"\bkar\s*rahi\b",
    r"\bho\s*rahi\b", r"\bthi\b",
    r"\bgayi\b", r"\baayi\b", r"\bboli\b", r"\bsoti\b",
    r"\bkhayi\b", r"\bbaithi\b",
    r"\bmata\b", r"\bbeti\b", r"\bdidi\b", r"\bmadam\b",
    r"\bauntie\b", r"\bbehan\b", r"\bbehen\b",
]

# Devanagari masculine markers (रहा, था, गया, आया, बोला, भाई, पिता, बेटा)
_HINDI_MALE_DEVA = [
    "रहा", "रहा हूं", "रहा हूँ", "रहा है", "रहे हैं",
    "था", "गया", "आया", "बोला", "सोया", "खाया", "बैठा",
    "भाई", "भैया", "पिता", "बेटा", "चाचा", "मामा",
]

# Devanagari feminine markers (रही, थी, गई, आई, बोली, दीदी, माता, बेटी)
_HINDI_FEMALE_DEVA = [
    "रही", "रही हूं", "रही हूँ", "रही है", "रही हैं",
    "थी", "गई", "आई", "बोली", "सोई", "खाई", "बैठी",
    "दीदी", "माता", "बेटी", "चाची", "मामी", "बहन",
    "मैडम", "महिला", "औरत", "लड़की",
]

# English pronouns
_EN_MALE = [r"\bhe\b", r"\bhim\b", r"\bhis\b", r"\bhimself\b"]
_EN_FEMALE = [r"\bshe\b", r"\bher\b", r"\bhers\b", r"\bherself\b"]

def _normalize(s: str) -> str:
    """NFC-normalize a string so Devanagari comparisons work regardless of source encoding."""
    return unicodedata.normalize("NFC", s)


# Pre-normalize all Devanagari marker lists once at import time
_HINDI_MALE_DEVA_NORM = [_normalize(m) for m in _HINDI_MALE_DEVA]
_HINDI_FEMALE_DEVA_NORM = [_normalize(m) for m in _HINDI_FEMALE_DEVA]


def _score_text(text: str) -> float:
    """Return a gender score for one utterance.

    Positive → male evidence, negative → female evidence.
    Handles both Devanagari (Deepgram language=hi output) and Romanized Latin.
    Primary signals: Hindi grammatical verb/adjective gender markers.
    Secondary signals: English pronouns.
    """
    text_norm = _normalize(text)
    t_lower = text_norm.lower()
    score = 0.0

    # ── Romanized Hindi grammatical markers (±0.4 each) ──
    for pat in _HINDI_MALE_ROMAN:
        if re.search(pat, t_lower):
            score += 0.4
    for pat in _HINDI_FEMALE_ROMAN:
        if re.search(pat, t_lower):
            score -= 0.4

    # ── Devanagari grammatical markers (±0.4 each) ──
    for marker in _HINDI_MALE_DEVA_NORM:
        if marker in text_norm:
            score += 0.4
    for marker in _HINDI_FEMALE_DEVA_NORM:
        if marker in text_norm:
            score -= 0.4

    # ── English pronouns (±0.3 each) ──
    for pat in _EN_MALE:
        if re.search(pat, t_lower):
            score += 0.3
    for pat in _EN_FEMALE:
        if re.search(pat, t_lower):
            score -= 0.3

    return score


def _label_and_confidence(cumulative: float, turns: int) -> tuple[GenderLabel, float]:
    """Convert a cumulative score into a label + confidence."""
    if turns == 0 or abs(cumulative) < 0.3:
        return "unknown", 0.0

    # Confidence grows with signal strength and number of turns, capped at 1.0
    raw_conf = min(abs(cumulative) / (1.0 + turns * 0.2), 1.0)
    # Round to 2 dp
    conf = round(raw_conf, 2)

    if cumulative > 0:
        return "male", conf
    else:
        return "female", conf


class GenderDetectionProcessor(FrameProcessor):
    """Pipecat processor that detects caller gender from transcriptions.

    Place it AFTER the STT service in the pipeline. It passes all frames
    through unchanged and emits a :class:`GenderDetectionFrame` downstream
    after every final transcription.

    Example::

        pipeline = Pipeline([
            transport.input(),
            stt,
            gender_detector,   # ← insert here
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ])

        @gender_detector.event_handler("on_gender_detected")
        async def on_gender_detected(processor, frame: GenderDetectionFrame):
            logger.info(f"Gender: {frame.gender} ({frame.confidence:.0%})")
    """

    DECAY = 0.5  # each old turn's weight halves every new turn

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cumulative_score: float = 0.0
        self._turns: int = 0
        self._register_event_handler("on_gender_detected")

    @property
    def current_gender(self) -> GenderLabel:
        label, _ = _label_and_confidence(self._cumulative_score, self._turns)
        return label

    @property
    def current_confidence(self) -> float:
        _, conf = _label_and_confidence(self._cumulative_score, self._turns)
        return conf

    def reset(self):
        """Reset accumulated state (e.g. on new call)."""
        self._cumulative_score = 0.0
        self._turns = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            turn_score = _score_text(frame.text)
            self._cumulative_score = self._cumulative_score * self.DECAY + turn_score
            self._turns += 1

            gender, confidence = _label_and_confidence(self._cumulative_score, self._turns)

            detection_frame = GenderDetectionFrame(
                gender=gender,
                confidence=confidence,
                turn_text=frame.text,
                cumulative_score=self._cumulative_score,
            )

            logger.debug(
                f"GenderDetection | turn={self._turns} | text='{frame.text}' | "
                f"turn_score={turn_score:+.2f} | cumulative={self._cumulative_score:+.2f} | "
                f"→ {gender} ({confidence:.0%})"
            )

            await self._call_event_handler("on_gender_detected", detection_frame)
            await self.push_frame(detection_frame, direction)

        await self.push_frame(frame, direction)
