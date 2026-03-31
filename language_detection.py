"""Language Detection Processor for Pipecat voice pipelines.

Analyzes each user transcription turn and maintains a running language estimate
with confidence score. Handles the India telephony context where callers switch
between Hindi (Devanagari), English, and code-mixed speech.

Detection strategy (layered):
  1. Script ratio — Devanagari characters → strong Hindi signal
                   Latin alphabetic chars  → English / Romanized-Hindi signal
  2. Romanized Hindi lexicon — unambiguous Romanized Hindi words in Latin text
     (e.g. "haan", "nahi", "kya") → code-mixed or Hindi signal
  3. Turn length heuristic — single-word / numbers-only responses get low
     confidence; label is still emitted so callers can see it update.

Dominant language is tracked across turns using exponential decay (DECAY=0.7)
so that a switch from Hindi to English is reflected quickly but noise/isolated
turns don't flip the dominant language instantly.

Emits a LanguageDetectionFrame after every final transcription.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


LanguageLabel = Literal["hindi", "english", "code-mixed", "other"]


@dataclass
class LanguageDetectionFrame(Frame):
    """Carries the language detection result after a transcription turn."""

    turn_language: LanguageLabel    # language of this specific turn
    dominant_language: LanguageLabel  # dominant language so far across all turns
    confidence: float               # 0.0 – 1.0, turn-level confidence
    dominant_confidence: float      # 0.0 – 1.0, dominant-language confidence
    turn_text: str                  # the transcript that triggered the update
    hindi_ratio: float              # fraction of alphabetic chars that are Devanagari
    english_ratio: float            # fraction of alphabetic chars that are Latin
    romanized_hits: int             # count of Romanized Hindi keywords found


# ── Romanized Hindi lexicon ────────────────────────────────────────────────
# Words that are unambiguously Romanized Hindi (won't appear in pure English).
# Kept as a set for O(1) lookup.
_ROMANIZED_HINDI_WORDS: set[str] = {
    # Affirmatives / negatives
    "haan", "han", "nahi", "nahin", "nai", "bilkul", "zaroor", "theek",
    "acha", "accha", "achha", "sahi",
    # Question words
    "kya", "kaun", "kaise", "kab", "kahan", "kyun", "kyunki",
    # Pronouns
    "mujhe", "mera", "meri", "mere", "aap", "aapko", "aapka", "aapki",
    "tumhara", "unka", "unki",
    # Common verbs / particles
    "bolo", "batao", "suno", "dekho", "karo", "karein", "chahiye",
    "chahta", "chahti", "chahte",
    # Frequently used connectors that don't overlap with English
    "lekin", "bahut", "thoda", "waise", "toh", "matlab", "samjhe",
    "samjha", "samjhi",
    # Hindi numerals (Romanized)
    "ek", "do", "teen", "chaar", "paanch", "chhah", "saat", "aath", "nau", "das",
    # Miscellaneous telephony-common words
    "baat", "boliye", "suniye", "jiyo", "ruko", "thoro",
}

# Romanized Hindi patterns that need regex (multi-word or morphological)
_ROMANIZED_HINDI_PATTERNS: list[str] = [
    r"\bkar\s*raha\b", r"\bkar\s*rahi\b", r"\bho\s*raha\b", r"\bho\s*rahi\b",
    r"\bkr?\s*raha\b", r"\bkr?\s*rahi\b",
]


# ── Character-level helpers ────────────────────────────────────────────────

def _is_devanagari(c: str) -> bool:
    return "\u0900" <= c <= "\u097F"


def _is_latin_alpha(c: str) -> bool:
    return c.isalpha() and ord(c) < 128


def _normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ── Per-turn analysis ──────────────────────────────────────────────────────

def _analyze_turn(text: str) -> dict:
    """Classify a single utterance and return a detail dict.

    Returns:
        {
            "language": LanguageLabel,
            "confidence": float,
            "hindi_ratio": float,
            "english_ratio": float,
            "romanized_hits": int,
            "word_count": int,
            "reason": str,     # human-readable classification reason
        }
    """
    text_norm = _normalize(text.strip())
    t_lower = text_norm.lower()

    # ── Character counts ──
    deva_count = sum(1 for c in text_norm if _is_devanagari(c))
    latin_count = sum(1 for c in text_norm if _is_latin_alpha(c))
    digit_count = sum(1 for c in text_norm if c.isdigit())
    total_alpha = deva_count + latin_count

    # Edge case: only numbers / punctuation / empty
    if total_alpha == 0:
        return {
            "language": "other",
            "confidence": 0.0,
            "hindi_ratio": 0.0,
            "english_ratio": 0.0,
            "romanized_hits": 0,
            "word_count": 0,
            "reason": "numbers_or_punctuation_only",
        }

    hindi_ratio = deva_count / total_alpha
    english_ratio = latin_count / total_alpha

    # ── Romanized Hindi signals (only relevant when Latin text present) ──
    latin_words = re.findall(r"\b[a-z]+\b", t_lower)
    word_count_latin = len(latin_words)

    romanized_hits = sum(1 for w in latin_words if w in _ROMANIZED_HINDI_WORDS)
    for pat in _ROMANIZED_HINDI_PATTERNS:
        if re.search(pat, t_lower):
            romanized_hits += 1

    # Approximate total word count (Devanagari words + Latin words)
    deva_words = len([w for w in text_norm.split() if any(_is_devanagari(c) for c in w)])
    total_words = deva_words + word_count_latin

    romanized_ratio = romanized_hits / max(word_count_latin, 1)

    # ── Classification rules ──
    if hindi_ratio >= 0.8:
        if english_ratio >= 0.15:
            lang = "code-mixed"
            reason = f"mostly_devanagari({hindi_ratio:.0%})_with_latin"
        else:
            lang = "hindi"
            reason = f"devanagari_dominant({hindi_ratio:.0%})"

    elif deva_count > 0 and latin_count > 0:
        # Mix — at least some of both scripts
        lang = "code-mixed"
        reason = f"mixed_scripts(hi={hindi_ratio:.0%},en={english_ratio:.0%})"

    elif english_ratio >= 0.8:
        # Pure Latin — check if it's Romanized Hindi
        if romanized_hits >= 2 or romanized_ratio >= 0.4:
            lang = "code-mixed"
            reason = f"romanized_hindi(hits={romanized_hits},ratio={romanized_ratio:.0%})"
        elif romanized_hits == 1 and total_words <= 3:
            lang = "code-mixed"
            reason = f"short_romanized_hindi(hits={romanized_hits})"
        else:
            lang = "english"
            reason = f"latin_dominant({english_ratio:.0%})"

    else:
        # Fallback — should rarely reach here
        lang = "english"
        reason = "fallback_latin"

    # ── Turn-level confidence ──
    # Script purity is the primary signal — Devanagari is unambiguous Hindi,
    # pure Latin is likely English. Word count adds a small bonus.
    if total_words == 0:
        confidence = 0.1
    elif lang == "hindi" and hindi_ratio >= 0.8:
        # Devanagari characters are unambiguous — high base even for 1 word
        base = 0.75
        confidence = round(min(base + (total_words - 1) * 0.05, 0.95), 2)
    elif lang == "english" and romanized_hits == 0:
        # Pure Latin, no Romanized Hindi — fairly clear signal
        base = 0.60
        confidence = round(min(base + (total_words - 1) * 0.05, 0.95), 2)
    elif lang == "code-mixed":
        # Mixed signals — moderate confidence, grows with length
        base = 0.55 if total_words >= 2 else 0.45
        confidence = round(min(base + total_words * 0.04, 0.90), 2)
    else:
        # Romanized Hindi detected in Latin text — moderate confidence
        base = 0.50 if total_words >= 2 else 0.40
        confidence = round(min(base + total_words * 0.04, 0.85), 2)

    return {
        "language": lang,
        "confidence": confidence,
        "hindi_ratio": round(hindi_ratio, 2),
        "english_ratio": round(english_ratio, 2),
        "romanized_hits": romanized_hits,
        "word_count": total_words,
        "reason": reason,
    }


# ── Dominant language helper ───────────────────────────────────────────────

_LANG_ORDER: list[LanguageLabel] = ["hindi", "english", "code-mixed", "other"]


def _dominant_language(scores: dict[str, float], turns: int) -> tuple[LanguageLabel, float]:
    """Return dominant language label and confidence from cumulative scores."""
    if turns == 0:
        return "other", 0.0

    best_lang: LanguageLabel = "other"
    best_score = 0.0
    total_score = sum(scores.values())

    for lang in _LANG_ORDER:
        if scores[lang] > best_score:
            best_score = scores[lang]
            best_lang = lang

    if total_score == 0 or best_score == 0:
        return "other", 0.0

    # Confidence = share of total signal × turn ramp-up.
    # raw_conf: how dominant the best language is vs. all accumulated signal.
    # turn_factor: ramps from 0.7 (turn 1) → 1.0 (turn 5+) so we're not
    # overconfident from a single utterance.
    raw_conf = min(best_score / max(total_score, 0.01), 1.0)
    turn_factor = min(0.7 + (turns - 1) * 0.075, 1.0)
    conf = round(raw_conf * turn_factor, 2)

    return best_lang, conf


# ── Processor ─────────────────────────────────────────────────────────────

class LanguageDetectionProcessor(FrameProcessor):
    """Pipecat processor that detects caller language from transcriptions.

    Place it AFTER the STT service in the pipeline. It passes all frames
    through unchanged and emits a :class:`LanguageDetectionFrame` downstream
    after every final transcription.

    Language is slower to switch than gender, so DECAY=0.7 (each old turn
    retains 70% weight), making the dominant language stable across the call
    while still reacting to a genuine switch within 3–4 turns.

    Example::

        pipeline = Pipeline([
            transport.input(),
            stt,
            language_detector,   # ← insert here
            gender_detector,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ])

        @language_detector.event_handler("on_language_detected")
        async def on_language_detected(processor, frame: LanguageDetectionFrame):
            logger.info(
                f"Language: {frame.turn_language} | "
                f"Dominant: {frame.dominant_language} ({frame.dominant_confidence:.0%})"
            )
    """

    # Language changes less frequently than gender in a conversation.
    # Higher decay → slower dominant-language updates, more stable output.
    DECAY = 0.7

    # Score added per turn for each language bucket.
    # Pure hindi/english turns don't credit code-mixed — that was causing
    # code-mixed to accumulate score on every turn and dilute dominant confidence.
    _TURN_WEIGHTS: dict[str, dict[str, float]] = {
        "hindi":      {"hindi": 1.0, "english": 0.0, "code-mixed": 0.0, "other": 0.0},
        "english":    {"hindi": 0.0, "english": 1.0, "code-mixed": 0.0, "other": 0.0},
        "code-mixed": {"hindi": 0.4, "english": 0.4, "code-mixed": 1.0, "other": 0.0},
        "other":      {"hindi": 0.0, "english": 0.0, "code-mixed": 0.0, "other": 0.1},
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scores: dict[str, float] = {"hindi": 0.0, "english": 0.0, "code-mixed": 0.0, "other": 0.0}
        self._turns: int = 0
        self._register_event_handler("on_language_detected")

    @property
    def dominant_language(self) -> LanguageLabel:
        lang, _ = _dominant_language(self._scores, self._turns)
        return lang

    @property
    def dominant_confidence(self) -> float:
        _, conf = _dominant_language(self._scores, self._turns)
        return conf

    def reset(self):
        """Reset accumulated state (e.g. on new call)."""
        self._scores = {"hindi": 0.0, "english": 0.0, "code-mixed": 0.0, "other": 0.0}
        self._turns = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            try:
                detail = _analyze_turn(frame.text)
            except Exception as exc:
                logger.warning(f"LanguageDetection | analysis failed: {exc}")
                await self.push_frame(frame, direction)
                return

            turn_lang: LanguageLabel = detail["language"]
            turn_conf: float = detail["confidence"]

            # ── Apply decay then accumulate ──
            weights = self._TURN_WEIGHTS[turn_lang]
            for lang in self._scores:
                self._scores[lang] = self._scores[lang] * self.DECAY + weights[lang] * turn_conf

            self._turns += 1

            dom_lang, dom_conf = _dominant_language(self._scores, self._turns)

            detection_frame = LanguageDetectionFrame(
                turn_language=turn_lang,
                dominant_language=dom_lang,
                confidence=turn_conf,
                dominant_confidence=dom_conf,
                turn_text=frame.text,
                hindi_ratio=detail["hindi_ratio"],
                english_ratio=detail["english_ratio"],
                romanized_hits=detail["romanized_hits"],
            )

            logger.debug(
                f"LanguageDetection | turn={self._turns} | text='{frame.text}' | "
                f"turn={turn_lang}({turn_conf:.0%}) | "
                f"dominant={dom_lang}({dom_conf:.0%}) | "
                f"hi={detail['hindi_ratio']:.0%} en={detail['english_ratio']:.0%} "
                f"rom_hits={detail['romanized_hits']} words={detail['word_count']} | "
                f"[{detail['reason']}]"
            )

            await self._call_event_handler("on_language_detected", detection_frame)
            await self.push_frame(detection_frame, direction)

        await self.push_frame(frame, direction)
