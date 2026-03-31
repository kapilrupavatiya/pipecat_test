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

# Hindi masculine verb/adjective endings (raha, tha, aaya, gaya, …)
_HINDI_MALE = [
    r"\bkr?\s*raha\b", r"\bkar\s*raha\b",
    r"\bho\s*raha\b", r"\btha\b", r"\btha\s*main\b",
    r"\bgaya\b", r"\baaya\b", r"\bbola\b", r"\bsota\b",
    r"\bkhaya\b", r"\bpiya\b", r"\bbaitha\b",
    r"\bmain\s+\w+\s*tha\b",
    r"\bpita\b", r"\bbeta\b", r"\bbhai\b", r"\bsir\b",
    r"\buncle\b",
]

# Hindi feminine verb/adjective endings (rahi, thi, aayi, gayi, …)
_HINDI_FEMALE = [
    r"\bkr?\s*rahi\b", r"\bkar\s*rahi\b",
    r"\bho\s*rahi\b", r"\bthi\b",
    r"\bgayi\b", r"\baayi\b", r"\bboli\b", r"\bsoti\b",
    r"\bkhayi\b", r"\bpiyi\b", r"\bbaithi\b",
    r"\bmain\s+\w+\s*thi\b",
    r"\bmata\b", r"\bbeti\b", r"\bdidi\b", r"\bmadam\b",
    r"\bauntie\b", r"\bbehan\b", r"\bbehen\b",
]

# English pronouns
_EN_MALE = [r"\bhe\b", r"\bhim\b", r"\bhis\b", r"\bhimself\b"]
_EN_FEMALE = [r"\bshe\b", r"\bher\b", r"\bhers\b", r"\bherself\b"]

# Common Indian first names (not exhaustive, but covers frequent ones)
_MALE_NAMES = {
    "rahul", "rohan", "raj", "rajesh", "amit", "arun", "arjun", "aman",
    "ankur", "ankit", "anil", "ajay", "akash", "akhil", "ashish", "ashok",
    "deepak", "dhruv", "gaurav", "harsh", "hemant", "ishan", "jay",
    "karan", "kartik", "kunal", "manoj", "mohit", "mukesh", "naveen",
    "neil", "nikhil", "nilesh", "pankaj", "prateek", "praveen", "puneet",
    "rajan", "rakesh", "ramesh", "ravi", "ritesh", "rohit", "sachin",
    "sahil", "sanjay", "sanjeev", "shubham", "siddharth", "sunil", "suresh",
    "tarun", "tushar", "varun", "vijay", "vikram", "vinay", "vishal",
    "vivek", "yash", "yogesh",
}
_FEMALE_NAMES = {
    "aditi", "aishwarya", "akanksha", "alka", "ananya", "anuradha",
    "aparna", "archana", "arjita", "astha", "bhavna", "deepa", "deepika",
    "divya", "ekta", "garima", "geeta", "harpreet", "heena", "ishika",
    "jyoti", "kajal", "kamya", "kavita", "khushi", "kirti", "komal",
    "kritika", "lata", "madhuri", "manisha", "megha", "meera", "monika",
    "naina", "namrata", "neha", "nidhi", "nisha", "palak", "poonam",
    "pooja", "prachi", "pragya", "priya", "priyanka", "radha", "rashmi",
    "rashi", "rekha", "renu", "rhea", "richa", "ritu", "riya", "rupal",
    "sakshi", "saloni", "sangeeta", "sapna", "seema", "shilpa", "shruti",
    "simran", "sneha", "sonam", "sonia", "srishti", "sunita", "swati",
    "tanvi", "tanya", "trisha", "usha", "varsha", "vidya", "vineeta",
}


def _score_text(text: str) -> float:
    """Return a gender score for one utterance.

    Positive → male evidence, negative → female evidence.
    """
    t = text.lower()
    score = 0.0

    # Hindi grammatical markers (strong signal, ±0.4 each)
    for pat in _HINDI_MALE:
        if re.search(pat, t):
            score += 0.4
    for pat in _HINDI_FEMALE:
        if re.search(pat, t):
            score -= 0.4

    # English pronouns (moderate, ±0.3)
    for pat in _EN_MALE:
        if re.search(pat, t):
            score += 0.3
    for pat in _EN_FEMALE:
        if re.search(pat, t):
            score -= 0.3

    # Name detection (strong signal, ±0.6 — cap at one per utterance)
    words = re.findall(r"\b[a-z]+\b", t)
    male_name_hit = any(w in _MALE_NAMES for w in words)
    female_name_hit = any(w in _FEMALE_NAMES for w in words)
    if male_name_hit and not female_name_hit:
        score += 0.6
    elif female_name_hit and not male_name_hit:
        score -= 0.6

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
            self._cumulative_score += turn_score
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
