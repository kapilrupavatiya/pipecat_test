#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Kotak Securities Shilpa Bot — Telephony WebSocket version.

Uses FastAPI WebSocket transport (AwaazAI serializer) with:
  - Deepgram STT (nova-2, language=hi, 8kHz telephony)
  - Groq LLM (llama-3.3-70b-versatile)
  - ElevenLabs TTS (eleven_turbo_v2_5)

Run the bot using::

    uv run bot_kotak.py
"""

import asyncio
import json
import os

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.responses import HTMLResponse

print("🚀 Starting Kotak bot (Telephony WebSocket)...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading LocalSmartTurnAnalyzerV3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ LocalSmartTurnAnalyzerV3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame

logger.info("Loading pipeline components...")
from deepgram import LiveOptions
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TranscriptionUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.observers.loggers.user_bot_latency_log_observer import UserBotLatencyLogObserver

from awaazde_serializer import AwaazAIFrameSerializer
from escalation_detection import EscalationDetectionProcessor, EscalationFrame
from freeswitch_esl import FreeSwitchESL
from gender_detection import GenderDetectionProcessor
from language_detection import LanguageDetectionProcessor

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

_WEBHOOK_URL = "https://webhook.site/34bbfede-7078-4d89-8a3b-3075b0e23ece"


async def _post_webhook(payload: dict) -> None:
    """Fire-and-forget POST to the webhook. Errors are logged but never raise."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(_WEBHOOK_URL, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                logger.debug(f"Webhook POST → {resp.status} | payload={payload}")
    except Exception as exc:
        logger.warning(f"Webhook POST failed: {exc}")


async def run_bot(transport: BaseTransport, call_uuid: str | None = None, handle_sigint: bool = False):
    logger.info(f"Starting bot | call_uuid={call_uuid}")

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-2",
            language="hi",
            smart_format=True,
            punctuate=True,
            encoding="linear16",
            channels=1,
            sample_rate=8000,
            endpointing=100,
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        model="eleven_turbo_v2_5",
        params=ElevenLabsTTSService.InputParams(
            language=Language.HI,
            auto_mode=True,
        ),
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        max_tokens=512,  # Voice replies are short — cap to reduce TTFB
    )

    messages = [
        {
            "role": "system",
            "content": """# LANGUAGE RULE — check before every reply
Look at the user's LATEST message only (ignore all prior turns):
- Contains ANY Devanagari character (अ, आ, क, ख, etc.)? → Reply 100% Hindi Devanagari. Zero Latin.
- No Devanagari? → Reply 100% English Latin. Zero Devanagari. No mixing. No exceptions.
Examples: "yes" → English | "ok sure" → English | "हाँ बोलिए" → Hindi | "जी हाँ" → Hindi

---

# Kotak Securities — Shilpa

You are Shilpa, a warm voice agent from Kotak Securities calling Neil to complete their De-mat account opening. Voice call rules: 1–3 sentences per reply, no lists or bullet points, one step at a time, always wait for response before advancing.

---

## Flow

### 1. GREETING
Confirm you're speaking with Neil.
- Confirmed (yes / haan / ji / speaking) → CALL PURPOSE
- Only "hello" or no reply → re-greet once
- Someone else answers → still go to CALL PURPOSE

### 2. CALL PURPOSE
Explain you're calling to help complete their De-mat account opening. Good time to talk?
- Busy → RESCHEDULE
- Agrees → CUSTOMER PROFILING

### 3. CUSTOMER PROFILING
Say you have a few quick questions to suggest the best plan. If they decline → briefly pitch Kotak Neo (quick digital process, trusted brand, competitive plans). If still no → end warmly.

Ask one at a time, in order:

a) "Have you ever invested in stocks, mutual funds, or IPOs?"

No → Ask if they've heard of the share market / NSE / BSE.
  - Yes → mention FDs/savings give limited returns; De-mat/MFs offer better growth → BEGINNER PITCH
  - No → BEGINNER PITCH directly

Yes → continue:
  - How often — daily or occasionally?
  - What — equity, derivatives, currency, commodity, or mutual funds?
  - Age?
  - Currently use Margin Trading Facility (MTF)?

Routing after profiling:
- Age < 30 → TRADE FREE YOUTH PLAN
- Age ≥ 30 + uses MTF → MTF ACTIVE USER
- Age ≥ 30 + no MTF + knows MTF → TRADE FREE PLAN
- Age ≥ 30 + no MTF + doesn't know MTF → MTF PITCH → TRADE FREE PRO PLAN

If asked why these questions → to suggest the most suitable plan.

---

## Plans

TRADE FREE PLAN — Zero brokerage first 30 days; ₹10 or 0.05% intraday (lower); 0.20% delivery; ₹10 per F&O order; free MF and IPO investing. Ask if they'd like to proceed.

TRADE FREE PRO PLAN — MTF at 9.69% p.a. vs ~18% market rate; intraday 0.05% or ₹10 (lower); delivery 0.10%; F&O ₹10/order; zero brokerage first 30 days. Ask if they'd like to proceed.

TRADE FREE YOUTH PLAN — No brokerage on stock delivery; intraday 0.05% or ₹10; F&O ₹10/order; account opening ₹99 (incl. GST); free MF and IPO investing. Mention you've sent a link to their registered number and will stay on the call to help.

BEGINNER PITCH — Reassure: everyone starts somewhere. Kotak has beginner tutorials, How-to videos, dedicated RM support for first 3 trades, and free research. Ask age → route to TRADE FREE PLAN or YOUTH PLAN.

MTF ACTIVE USER — Kotak MTF at 9.69% p.a. (0.027%/day) vs ~18%; up to 4x leverage on 1,000+ stocks; no holding time limit; MTF ideas in Kotak Neo app. Suggest Trade Free Pro Plan.

MTF PITCH — MTF means borrowing from broker to buy more stocks, up to 4x on eligible stocks. Most brokers charge ~18% p.a.; Kotak charges 9.69% on the Pro Plan. Then → TRADE FREE PRO PLAN.

---

## Account Opening
Tell Neil to open the Kotak Neo app and enter their name and registered mobile number. Stay on the call to help.
Steps (share only if asked): enter mobile and email and verify OTP, then upload PAN and Aadhaar (Aadhaar must be linked to mobile), then wait for documents to be reviewed and account to be activated, then login credentials arrive via email and SMS.
- Wants to continue later → RESCHEDULE
- Waiting for app response → tell them to wait for email confirmation and explore the app meanwhile.

---

## Reschedule
Ask what day works. Then ask for a time if not given. Confirm the slot and end warmly.
If they decline or don't specify → say you'll call at a convenient time. Have a great day.

---

## FAQ (answer briefly, then return to flow)
- Stock market? — Buy/sell company shares; value grows when the company does well.
- Why Kotak? — 30+ years, 5 million+ customers, ₹10 max brokerage vs ₹20 elsewhere, dedicated support.
- Safe? — SEBI regulated, bank-grade security, backed by Kotak Group.
- Min investment? — SIPs from ₹500; no minimum balance to open account.
- De-mat? — Digital locker for shares; needed to buy/sell stocks.
- De-mat vs Trading? — De-mat stores shares; Trading account buys/sells. Both opened together at Kotak.
- Documents? — PAN + Aadhaar (linked to mobile). That's it.
- How long? — 5–10 mins. Active in 24–48 hours after verification.
- Charges? — ~₹300–400/year AMC if portfolio is above ₹50,000.
- Tax? — STCG 20%, LTCG above ₹1 lakh at 12.5%. P&L reports in Kotak Neo.
- Another broker? — Many customers have multiple; our MTF at 9.69% and dedicated support are hard to match.
- Closing? — No lock-in. Close anytime.
- Recommendations? — Free inside Kotak Neo after account activation.
- Send link/details? — Will send to registered number after the call; ask if they want to continue now.

---

## Special Cases
- Who are you? → "I'm Shilpa from Kotak Securities, calling to help with your De-mat account opening."
- Are you AI? → "Yes, I'm an AI assistant from Kotak Securities. Happy to help — shall we continue?"
- Stop calling → "Okay, thank you for your time. Have a great day!"
- Safety / mental health → Empathy in 1–2 sentences, suggest professional help, end gently.
- Existing Kotak customer → KYC already done, will be even faster.
- No response 2 turns in a row → Thank them and end the call.
- Other Kotak services → Note the request, say relevant team will reach out, then continue.

Always end any closing with "Have a great day!"

*Today's date: March 31, 2026 (IST)*""",
        },
    ]

    context = LLMContext(messages)

    language_detector = LanguageDetectionProcessor()

    @language_detector.event_handler("on_language_detected")
    async def on_language_detected(processor, frame):
        logger.debug(
            f"🌐 Language turn: {frame.turn_language}({frame.confidence:.0%}) | "
            f"dominant: {frame.dominant_language}({frame.dominant_confidence:.0%}) | "
            f"text='{frame.turn_text}'"
        )
        asyncio.create_task(_post_webhook({
            "event": "language_detected",
            "turn_language": frame.turn_language,
            "turn_confidence": round(frame.confidence, 2),
            "dominant_language": frame.dominant_language,
            "dominant_confidence": round(frame.dominant_confidence, 2),
            "hindi_ratio": frame.hindi_ratio,
            "english_ratio": frame.english_ratio,
            "romanized_hits": frame.romanized_hits,
            "turn_text": frame.turn_text,
        }))

    # ── Escalation detector ───────────────────────────────────────────────
    escalation_detector = EscalationDetectionProcessor()

    @escalation_detector.event_handler("on_escalation_triggered")
    async def on_escalation_triggered(processor, frame: EscalationFrame):
        logger.warning(
            f"🚨 Escalation triggered | reason={frame.reason} | "
            f"turn={frame.turn_number} | frustration={frame.frustration_score} | "
            f"text='{frame.trigger_text}'"
        )

        # 1. Log to webhook (fire-and-forget)
        asyncio.create_task(_post_webhook({
            "event": "escalation_triggered",
            "reason": frame.reason,
            "turn_number": frame.turn_number,
            "trigger_text": frame.trigger_text,
            "frustration_score": frame.frustration_score,
            "transcript": frame.transcript,
            "call_uuid": call_uuid,
        }))

        # 2. Choose appropriate handover message for the LLM
        _reason_instructions = {
            "explicit_request": (
                "The caller just asked to speak with a human agent. "
                "Warmly acknowledge their request, say you are transferring them "
                "to a human agent right now, and wish them well. "
                "Keep it to 1–2 sentences. Apply the language rule."
            ),
            "frustration": (
                "The caller seems frustrated. Apologise briefly and say "
                "you are connecting them to a human agent who can better assist. "
                "Keep it to 1–2 sentences. Apply the language rule."
            ),
            "loop": (
                "The conversation seems stuck. Say you understand this is taking "
                "longer than expected and you will connect them to a human agent "
                "for further assistance. Keep it to 1–2 sentences. Apply the language rule."
            ),
            "out_of_domain": (
                "The caller's question is outside your scope. "
                "Politely say you'll connect them to a specialist who can help, "
                "and that you are transferring them now. "
                "Keep it to 1–2 sentences. Apply the language rule."
            ),
        }
        instruction = _reason_instructions.get(
            frame.reason,
            "Transfer the caller to a human agent. Say you are doing so now. "
            "Keep it to 1 sentence. Apply the language rule.",
        )

        await escalation_detector.push_frame(
            LLMMessagesAppendFrame(messages=[{"role": "system", "content": instruction}])
        )
        await task.queue_frames([LLMRunFrame()])

        # 3. Bridge caller to human agent after TTS finishes speaking (~5 s)
        async def _do_transfer():
            await asyncio.sleep(2)
            if call_uuid:
                try:
                    esl = FreeSwitchESL()
                    result = await esl.bridge_to_agent(call_uuid)
                    logger.info(f"FreeSWITCH bridge result: {result!r}")
                except Exception as exc:
                    logger.error(f"FreeSWITCH bridge failed: {exc}")
            else:
                logger.warning("No call_uuid available — cannot bridge to human agent")
            await task.cancel()

        asyncio.create_task(_do_transfer())

    gender_detector = GenderDetectionProcessor()
    _injected_gender = None

    @gender_detector.event_handler("on_gender_detected")
    async def on_gender_detected(processor, frame):
        nonlocal _injected_gender

        logger.debug(
            f"👤 Gender turn: {frame.gender} ({frame.confidence:.0%}) | "
            f"cumulative={frame.cumulative_score:+.2f} | text='{frame.turn_text}'"
        )
        asyncio.create_task(_post_webhook({
            "event": "gender_detected",
            "gender": frame.gender,
            "confidence": round(frame.confidence, 2),
            "cumulative_score": round(frame.cumulative_score, 3),
            "turn_text": frame.turn_text,
        }))

        # Inject into LLM context only when gender changes with sufficient confidence
        if frame.confidence >= 0.5 and frame.gender != "unknown" and frame.gender != _injected_gender:
            _injected_gender = frame.gender

            if frame.gender == "male":
                gender_note = (
                    "CALLER GENDER UPDATE: The caller has been identified as MALE. "
                    "In English, address them as 'sir'. "
                    "In Hindi, use masculine verb/adjective forms "
                    "(e.g., 'kar rahe hain', 'tha', 'raha hoon'). "
                    "Never use 'madam' or feminine Hindi forms like 'rahi', 'thi'."
                )
            else:
                gender_note = (
                    "CALLER GENDER UPDATE: The caller has been identified as FEMALE. "
                    "In English, address them as 'ma'am'. "
                    "In Hindi, use feminine verb/adjective forms "
                    "(e.g., 'kar rahi hain', 'thi', 'rahi hoon'). "
                    "Never use 'sir' or masculine Hindi forms like 'raha', 'tha'."
                )

            await gender_detector.push_frame(
                LLMMessagesAppendFrame(messages=[{"role": "system", "content": gender_note}])
            )
            logger.info(
                f"👤 Gender injected into LLM: {frame.gender} ({frame.confidence:.0%})"
            )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[TranscriptionUserTurnStopStrategy()],
            )
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            language_detector,
            escalation_detector,
            gender_detector,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=False,
            enable_metrics=True,
            enable_usage_metrics=True,
            observers=[UserBotLatencyLogObserver()],
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def start_call():
    print("POST TwiML")
    try:
        with open("templates/streams.xml", "r") as f:
            content = f.read()
    except FileNotFoundError:
        content = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://your-server.com/ws" />
    </Connect>
</Response>"""
    return HTMLResponse(content=content, media_type="application/xml")


@app.websocket("/ws/21")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    print(call_data, flush=True)
    stream_sid = call_data["start"]["stream_sid"]
    # FreeSWITCH call UUID — prefer explicit call_sid if present, fall back to stream_sid
    call_uuid = (
        call_data["start"].get("call_sid")
        or call_data["start"].get("uuid")
        or stream_sid
    )
    print("WebSocket connection accepted")

    serializer = AwaazAIFrameSerializer(stream_sid=stream_sid)

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=8000,
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.2,
                    stop_secs=0.2,
                    min_volume=0.6,
                ),
            ),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
            vad_audio_passthrough=True,
            serializer=serializer,
        ),
    )

    await run_bot(transport, call_uuid=call_uuid, handle_sigint=False)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8770)
