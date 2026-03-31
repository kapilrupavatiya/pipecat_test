#
# Call Report Observer for Pipecat pipelines.
#
# Tracks turn-by-turn metrics (STT, NLU/LLM, TTS) across a call and saves:
#   - Per-turn user audio WAV files
#   - Per-turn bot audio WAV files
#   - A JSON report with latencies, transcripts, and file paths
#

"""Turn-by-turn call report observer for Pipecat pipelines."""

import json
import os
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


@dataclass
class TurnReport:
    """Data collected for a single conversation turn."""

    turn_number: int

    # STT
    stt_request_id: str = ""          # STT provider session/request ID (e.g. dg-request-id)
    stt_audio_path: str = ""          # path to saved user audio WAV
    stt_latency_ms: float = 0.0       # ms from user stopped speaking → final transcript
    stt_output: str = ""              # final transcript
    stt_intermediate_outputs: str = ""  # comma-separated interim transcripts

    # NLU / LLM
    nlu_input: str = ""               # last user message sent to LLM
    nlu_latency_ms: float = 0.0       # ms from LLMFullResponseStart → LLMFullResponseEnd
    nlu_output: str = ""              # full LLM text response

    # TTS
    tts_input: str = ""               # text sent to TTS
    tts_latency_ms: float = 0.0       # ms from TTSStarted → TTSStopped
    tts_audio_path: str = ""          # path to saved bot audio WAV

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CallReportObserver(BaseObserver):
    """Pipeline observer that builds a turn-by-turn call report.

    For each conversation turn it records:
    - STT: user audio file, latency, final transcript, intermediate transcripts
    - NLU: LLM input, latency, LLM output
    - TTS: input text, latency, bot audio file

    At the end of the call a JSON report and per-turn WAV files are saved
    under ``report_dir/<YYYYMMDD_HHMMSS>/``.

    Design notes:
    - Pipecat pushes every frame once per pipeline hop (N processors = N calls).
      We deduplicate using the frame object's identity (id(frame)) so each frame
      is only processed once regardless of how many hops it travels.
    - A turn is only closed on BotStoppedSpeakingFrame once the turn has received
      an STT transcript, preventing premature closure from the bot's initial
      greeting (which fires BotStoppedSpeakingFrame before any user turn starts).

    Usage::

        observer = CallReportObserver(report_dir="reports")
        task = PipelineTask(
            pipeline,
            params=PipelineParams(observers=[observer]),
        )
    """

    def __init__(
        self,
        report_dir: str = "reports",
        stt_request_id_getter: Optional[Callable[[], Optional[str]]] = None,
    ):
        super().__init__()
        self._call_start = datetime.now()
        self._session_dir = os.path.join(
            report_dir, self._call_start.strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self._session_dir, exist_ok=True)

        self._turns: List[TurnReport] = []
        self._current_turn: Optional[TurnReport] = None
        self._turn_number = 0

        # ── Frame deduplication ───────────────────────────────────────────
        # Each Pipecat frame object travels through N processor hops, causing
        # on_push_frame to fire N times for the same logical event. We track
        # frame.id (a unique integer assigned at construction via obj_id())
        # to process each frame exactly once. Using Python's id() (memory
        # address) is unsafe because GC'd audio frame addresses get reused.
        self._seen_ids: Set[int] = set()

        # ── Turn gate: only close on BotStopped if user actually spoke ────
        # Prevents the bot's initial greeting from closing a user turn
        # that hasn't received any transcript yet.
        self._turn_has_transcript: bool = False

        # ── User audio buffering (between VADStart and VADStop) ────────────
        self._user_audio: List[bytes] = []
        self._user_audio_rate: int = 16000
        self._user_audio_ch: int = 1
        self._collecting_user_audio: bool = False

        # ── TTS audio buffering (between TTSStarted and TTSStopped) ───────
        self._tts_audio: List[bytes] = []
        self._tts_audio_rate: int = 16000
        self._tts_audio_ch: int = 1
        self._collecting_tts_audio: bool = False

        # ── Timing checkpoints ────────────────────────────────────────────
        self._stt_stop_time: Optional[float] = None   # monotonic, set at VADStop
        self._llm_start_time: Optional[float] = None  # monotonic, set at LLMFullResponseStart
        self._tts_start_time: Optional[float] = None  # monotonic, set at TTSStarted

        self._stt_request_id_getter = stt_request_id_getter

        logger.info(f"CallReportObserver: session dir → {self._session_dir}")

    # ── BaseObserver interface ─────────────────────────────────────────────

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame

        # ── Deduplication: skip frames we've already processed ─────────────
        # Every frame has a unique frame.id (integer, assigned at construction).
        # The same frame object is pushed once per pipeline hop, so we process
        # each unique frame exactly once regardless of hop count.
        fid = frame.id
        if fid in self._seen_ids:
            return
        self._seen_ids.add(fid)

        # ── User Speech / STT ──────────────────────────────────────────────
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._begin_turn()

        elif isinstance(frame, InputAudioRawFrame):
            if self._collecting_user_audio and self._current_turn:
                self._user_audio.append(frame.audio)
                self._user_audio_rate = frame.sample_rate
                self._user_audio_ch = frame.num_channels

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._current_turn:
                self._collecting_user_audio = False
                self._stt_stop_time = time.monotonic()
                path = self._save_wav(
                    self._user_audio,
                    self._user_audio_rate,
                    self._user_audio_ch,
                    f"turn_{self._turn_number:02d}_user.wav",
                )
                self._current_turn.stt_audio_path = path
                self._user_audio = []

        elif isinstance(frame, InterimTranscriptionFrame):
            if self._current_turn and frame.text:
                prev = self._current_turn.stt_intermediate_outputs
                self._current_turn.stt_intermediate_outputs = (
                    f"{prev}, {frame.text}" if prev else frame.text
                )

        elif isinstance(frame, TranscriptionFrame):
            if self._current_turn:
                self._current_turn.stt_output = frame.text
                self._turn_has_transcript = True
                if self._stt_stop_time is not None:
                    self._current_turn.stt_latency_ms = (
                        time.monotonic() - self._stt_stop_time
                    ) * 1000
                    self._stt_stop_time = None
                # Default NLU input to the transcript; may be overridden by
                # LLMContextFrame below if the context can be parsed
                self._current_turn.nlu_input = frame.text

        # ── LLM / NLU ─────────────────────────────────────────────────────
        elif isinstance(frame, LLMContextFrame):
            if self._current_turn:
                try:
                    msgs = frame.context.get_messages()
                    user_msgs = [m for m in msgs if m.get("role") == "user"]
                    if user_msgs:
                        content = user_msgs[-1].get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                c.get("text", "")
                                for c in content
                                if isinstance(c, dict)
                            )
                        self._current_turn.nlu_input = str(content)
                except Exception as exc:
                    logger.debug(
                        f"CallReportObserver: LLMContextFrame parse error: {exc}"
                    )

        elif isinstance(frame, LLMFullResponseStartFrame):
            if self._current_turn:
                self._llm_start_time = time.monotonic()

        elif isinstance(frame, LLMTextFrame):
            if self._current_turn and frame.text:
                self._current_turn.nlu_output += frame.text

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._current_turn and self._llm_start_time is not None:
                self._current_turn.nlu_latency_ms = (
                    time.monotonic() - self._llm_start_time
                ) * 1000
                self._llm_start_time = None

        # ── TTS ───────────────────────────────────────────────────────────
        elif isinstance(frame, TTSTextFrame):
            if self._current_turn and frame.text:
                self._current_turn.tts_input += frame.text

        elif isinstance(frame, TTSStartedFrame):
            if self._current_turn:
                self._tts_start_time = time.monotonic()
                self._collecting_tts_audio = True

        elif isinstance(frame, TTSAudioRawFrame):
            if self._collecting_tts_audio and self._current_turn:
                self._tts_audio.append(frame.audio)
                self._tts_audio_rate = frame.sample_rate
                self._tts_audio_ch = frame.num_channels

        elif isinstance(frame, TTSStoppedFrame):
            if self._current_turn:
                self._collecting_tts_audio = False
                if self._tts_start_time is not None:
                    self._current_turn.tts_latency_ms = (
                        time.monotonic() - self._tts_start_time
                    ) * 1000
                    self._tts_start_time = None

        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Only close the turn if the user actually spoke and we have a
            # transcript. This guards against the bot's initial greeting
            # (BotStoppedSpeakingFrame) incorrectly ending an empty user turn.
            if self._current_turn and self._turn_has_transcript:
                path = self._save_wav(
                    self._tts_audio,
                    self._tts_audio_rate,
                    self._tts_audio_ch,
                    f"turn_{self._turn_number:02d}_bot.wav",
                )
                self._current_turn.tts_audio_path = path
                self._tts_audio = []
                self._end_turn()

        elif isinstance(frame, (EndFrame, CancelFrame)):
            # EndFrame = normal shutdown; CancelFrame = client disconnect / task.cancel()
            self._generate_report()

    # ── Private helpers ────────────────────────────────────────────────────

    def _begin_turn(self):
        """Open a new conversation turn."""
        if self._current_turn and self._turn_has_transcript:
            # Previous turn never got BotStopped — close it before starting next
            self._end_turn()
        elif self._current_turn:
            # Previous turn had no transcript (e.g. false VAD trigger) — discard
            logger.debug(
                f"CallReportObserver: discarding empty turn {self._turn_number}"
            )
            self._current_turn = None

        self._turn_number += 1
        self._current_turn = TurnReport(turn_number=self._turn_number)
        if self._stt_request_id_getter:
            self._current_turn.stt_request_id = self._stt_request_id_getter() or ""
        self._turn_has_transcript = False
        self._collecting_user_audio = True
        self._user_audio = []
        self._tts_audio = []
        logger.info(
            f"CallReportObserver: ▶ Turn {self._turn_number} started"
            + (f" [stt_request_id={self._current_turn.stt_request_id}]" if self._current_turn.stt_request_id else "")
        )

    def _end_turn(self):
        """Close and store the current turn."""
        if not self._current_turn:
            return
        t = self._current_turn
        self._turns.append(t)
        logger.info(
            f"CallReportObserver: ■ Turn {t.turn_number} done — "
            f"STT {t.stt_latency_ms:.0f} ms | "
            f"NLU {t.nlu_latency_ms:.0f} ms | "
            f"TTS {t.tts_latency_ms:.0f} ms"
        )
        self._current_turn = None
        self._turn_has_transcript = False

    def _save_wav(
        self, chunks: List[bytes], rate: int, channels: int, filename: str
    ) -> str:
        """Write PCM chunks to a WAV file; return the saved path (or '' on error)."""
        if not chunks:
            return ""
        path = os.path.join(self._session_dir, filename)
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit PCM / S16LE
                wf.setframerate(rate)
                wf.writeframes(b"".join(chunks))
            logger.debug(f"CallReportObserver: saved {filename}")
        except Exception as exc:
            logger.error(f"CallReportObserver: WAV save failed ({filename}): {exc}")
            return ""
        return path

    def _generate_report(self):
        """Finalize any open turn and write the JSON report to disk."""
        # Save whatever was collected even if the turn wasn't cleanly closed
        # (e.g. call cancelled mid-turn)
        if self._current_turn and self._turn_has_transcript:
            if self._tts_audio:
                path = self._save_wav(
                    self._tts_audio,
                    self._tts_audio_rate,
                    self._tts_audio_ch,
                    f"turn_{self._turn_number:02d}_bot.wav",
                )
                self._current_turn.tts_audio_path = path
                self._tts_audio = []
            self._end_turn()

        report = {
            "call_start": self._call_start.isoformat(),
            "total_turns": len(self._turns),
            "turns": [t.to_dict() for t in self._turns],
        }

        report_path = os.path.join(self._session_dir, "call_report.json")
        try:
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
            logger.info(f"CallReportObserver: report → {report_path}")
        except Exception as exc:
            logger.error(f"CallReportObserver: report save failed: {exc}")

        # ── Console summary ────────────────────────────────────────────────
        sep = "=" * 64
        logger.info(sep)
        logger.info(
            f"CALL REPORT — {len(self._turns)} turns  "
            f"[{self._call_start.strftime('%Y-%m-%d %H:%M:%S')}]"
        )
        logger.info(sep)
        for t in self._turns:
            nlu_preview = (
                t.nlu_output[:80] + "…" if len(t.nlu_output) > 80 else t.nlu_output
            )
            tts_preview = (
                t.tts_input[:60] + "…" if len(t.tts_input) > 60 else t.tts_input
            )
            logger.info(f"  Turn {t.turn_number}:")
            if t.stt_request_id:
                logger.info(f"    STT  req id  : {t.stt_request_id}")
            logger.info(f"    STT  latency : {t.stt_latency_ms:.0f} ms")
            logger.info(f"    STT  output  : {t.stt_output!r}")
            logger.info(f"    STT  interim : {t.stt_intermediate_outputs or '—'}")
            logger.info(f"    NLU  input   : {t.nlu_input!r}")
            logger.info(f"    NLU  latency : {t.nlu_latency_ms:.0f} ms")
            logger.info(f"    NLU  output  : {nlu_preview!r}")
            logger.info(f"    TTS  input   : {tts_preview!r}")
            logger.info(f"    TTS  latency : {t.tts_latency_ms:.0f} ms")
            logger.info(f"    User audio   : {t.stt_audio_path or '—'}")
            logger.info(f"    Bot  audio   : {t.tts_audio_path or '—'}")
        logger.info(sep)
