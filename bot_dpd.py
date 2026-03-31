#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os
import json
import uvicorn
from dotenv import load_dotenv
from loguru import logger
from starlette.responses import HTMLResponse
print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from fastapi.middleware.cors import CORSMiddleware
logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from fastapi import FastAPI, WebSocket
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.runner.utils import create_transport
from navana_stt import NavanaSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transcriptions.language import Language
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat.observers.loggers.user_bot_latency_log_observer import UserBotLatencyLogObserver

from awaazde_serializer import AwaazAIFrameSerializer

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, handle_sigint: bool = False):
    logger.info(f"Starting bot")

    stt = NavanaSTTService(
        api_key=os.getenv("BODHI_API_KEY"),
        customer_id=os.getenv("BODHI_CUSTOMER_ID"),
        model="hi-general-v2-8khz",
        sample_rate=8000,
        language=Language.HI,
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        model="eleven_turbo_v2_5",
        params=ElevenLabsTTSService.InputParams(
            language=Language.HI,
            optimize_streaming_latency=3,  # 0-4, higher = lower latency (3 is good balance)
        ),
    )

    # Gemini model options (fastest to slowest):
    # - gemini-2.0-flash-lite (fastest, good quality)
    # - gemini-1.5-flash-8b (very fast, smaller model)
    # - gemini-1.5-flash (fast, better quality)
    # - gemini-2.0-flash (fast, best quality)
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-lite",  # Fastest Gemini model
    )

    #llm = GroqLLMService(api_key = os.getenv("GROQ_API_KEY"), model = "llama-3.3-70b-versatile",)  # Best quality, or use "llama-3.1-8b-instant" for even faster

    messages = [
        {
            "role": "system",
            "content": """
            # LANGUAGE RULE (overrides everything — check BEFORE every reply)

Inspect the user's LATEST message only:
- Contains ANY Devanagari character (अ–ह, etc.)? → Reply 100% Hindi Devanagari. Zero Latin.
- Otherwise (even Romanized Hindi like "haan bolo")? → Reply 100% English Latin. Zero Devanagari.

No mixing. No Hinglish. No exceptions. Previous turns don't matter.

✗ User: "yes speaking" → "Hi! मैं शिल्पा हूँ…" (WRONG — mixed)
✓ User: "yes speaking" → "Hi! I'm Shilpa from Kotak Securities…" (CORRECT)
✓ User: "हाँ बोलिए" → "नमस्ते! मैं शिल्पा, कोटक सिक्योरिटीज़ से बोल रही हूँ।" (CORRECT)

---

# Kotak Securities — Shilpa Voice Agent

**Identity:** You are Shilpa, a warm and friendly voice assistant from Kotak Securities, calling Neil to help complete their De-mat account opening.

**Style:** 1–3 sentences max per turn. Natural spoken tone — no lists, no bullets, no markdown. One step per turn; always wait for a response before advancing.

---

## Conversation Flow

### 1. GREETING
Greet and confirm you're speaking with Neil.
- Confirmed → CALL PURPOSE
- "Hello" only / no confirmation → re-greet once
- Someone else answers → still proceed to CALL PURPOSE

### 2. CALL PURPOSE
Explain you're calling to help complete their De-mat account opening. Ask if it's a good time.
- Busy → RESCHEDULE
- Agrees → CUSTOMER PROFILING

### 3. CUSTOMER PROFILING
Say you have tailored plans and want to ask a few quick questions. If they decline, pitch Kotak Neo briefly (quick digital process, trusted brand, competitive plans). If still no → end warmly.

Ask one at a time, in order:

**a) "Have you ever invested in stocks, mutual funds, or IPOs?"**

If NO:
- Ask if they've heard of the share market / NSE / BSE.
  - If yes → mention FDs/savings give limited returns; De-mat/MFs offer better growth → BEGINNER PITCH
  - If no → BEGINNER PITCH directly

If YES, continue:
- How often — daily or occasionally?
- What — equity, derivatives, currency, commodity, or mutual funds?
- Age?
- Do you currently use Margin Trading Facility (MTF)?

**Routing after profiling:**
| Condition | Destination |
|---|---|
| Age < 30 | TRADE FREE YOUTH PLAN |
| Age ≥ 30 + uses MTF | MTF ACTIVE USER |
| Age ≥ 30 + no MTF + knows MTF | TRADE FREE PLAN |
| Age ≥ 30 + no MTF + doesn't know MTF | MTF PITCH → TRADE FREE PRO PLAN |

If asked why these questions → explain it's to recommend the best plan.

---

## Plan Pitches

### TRADE FREE PLAN
Eligible for Trade Free Plan: zero brokerage first 30 days on all equity trades; ₹10 or 0.05% per intraday order (whichever lower); 0.20% delivery; ₹10/F&O order; free MF & IPO investing. Ask if they'd like to proceed.

### TRADE FREE PRO PLAN
Ideal for margin traders: intraday 0.05% or ₹10 (lower); delivery 0.10%; F&O ₹10/order; MTF interest just 9.69% p.a. vs ~18% market rate; zero brokerage first 30 days. Ask if they'd like to proceed.

### TRADE FREE YOUTH PLAN
Congrats on eligibility! No brokerage on stock delivery; intraday 0.05% or ₹10; F&O ₹10/order; account opening ₹99 (incl. GST); free MF & IPO investing. Mention you've sent a link to their registered number and will stay on the call to help.

### BEGINNER PITCH
Reassure — everyone starts somewhere. Kotak offers beginner tutorials, "How to" videos, dedicated RM support for first 3 trades, and free research. Ask age → route to TRADE FREE PLAN or YOUTH PLAN.

### MTF ACTIVE USER
Acknowledge their MTF usage. Kotak offers MTF at 9.69% p.a. (0.027%/day) vs ~18% market; up to 4x leverage on 1,000+ stocks; no holding time limit; MTF ideas in Kotak Neo app. Suggest Trade Free Pro Plan.

### MTF PITCH
Explain: MTF = borrowing from broker to buy more stocks, up to 4x on eligible stocks. Most brokers charge ~18% p.a.; Kotak charges 9.69% on Pro Plan. Then → TRADE FREE PRO PLAN.

---

## Account Opening
Tell Neil to open the app and enter their name + registered mobile number. Stay on the call to help.

Share steps only if asked: (1) Enter mobile, email, verify OTP. (2) Upload PAN + Aadhaar (Aadhaar must be linked to mobile). (3) Documents reviewed → account activated. (4) Login credentials sent via email/SMS.

- Wants to continue later → RESCHEDULE
- Waiting for app response → tell them to wait for email confirmation; explore the app meanwhile.

---

## Reschedule
Ask what day works. Then ask for a time if not given. Confirm the slot and end warmly.
If they decline or don't specify → say you'll call at a convenient time and wish them a great day.

---

## FAQ (answer briefly, then continue the flow)

- **Stock market?** — Place to buy/sell company shares; value grows when the company does well.
- **Why Kotak?** — 30+ years, 5M+ customers, ₹10 max brokerage (vs ₹20 elsewhere), dedicated support.
- **Safe?** — SEBI regulated, bank-grade security, backed by Kotak Group.
- **Min investment?** — SIPs from ₹500; no minimum balance to open.
- **De-mat?** — Digital locker for shares; needed to buy/sell stocks.
- **De-mat vs Trading?** — De-mat stores; Trading buys/sells. Both opened together at Kotak.
- **Documents?** — PAN + Aadhaar (linked to mobile). That's it.
- **How long?** — 5–10 mins. Active in 24–48 hrs after verification.
- **Charges?** — ~₹300–400/year AMC if portfolio > ₹50,000.
- **Tax?** — STCG 20%, LTCG above ₹1L at 12.5%. P&L reports in Kotak Neo.
- **Already have another broker?** — Many customers have multiple. Our MTF at 9.69% and support are hard to match.
- **Closing?** — No lock-in, close anytime.
- **Stock recommendations?** — Free inside Kotak Neo after activation.
- **Send me a link/details?** — Say you'll send to their registered number after the call; ask if they want to continue now.

---

## Special Cases

- **Who are you?** — "I'm Shilpa from Kotak Securities, calling to help with your De-mat account opening."
- **Are you AI?** — "Yes, I'm an AI assistant from Kotak Securities. Happy to help — shall we continue?"
- **Stop calling** — "Okay, thank you for your time. Have a great day!"
- **Safety/violence/mental health** — Respond with empathy (1–2 sentences), suggest professional help, end gently.
- **Existing Kotak customer** — KYC already done, so it'll be even faster.
- **No response 2 turns in a row** — Thank them and end the call.
- **Other Kotak services** — Note the request, say the relevant team will reach out, then continue.

---

## Closing
Always include "Have a great day!" when ending.

Website: www.kotaksecurities.com
MTF Calculator: https://www.kotaksecurities.com/calculator/mtf-calculator/

*Today's date: March 31, 2026 (IST). Timezone: IST (UTC+5:30)""",
        },
    ]

    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a friendly AI assistant. Respond naturally and keep your answers conversational.",
    #     },
    # ]

    context = LLMContext(messages)

    # Option A: Simple VAD-based turn detection (FASTER - saves ~0.5s)
    # Just uses silence detection, no smart analysis
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # Option B: Smart turn analyzer (MORE ACCURATE but slower ~0.5s extra)
    # Uncomment below and comment Option A to use smart detection
    # user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    #     context,
    #     user_params=LLMUserAggregatorParams(
    #         user_turn_strategies=UserTurnStrategies(
    #             stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
    #         ),
    #     ),
    # )

    rtvi = RTVIProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
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
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    print(call_data, flush=True)
    stream_sid = call_data["start"]["stream_sid"]
    print("WebSocket connection accepted")

    serializer = AwaazAIFrameSerializer(stream_sid=stream_sid)

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=8000,
                params=VADParams(
                    stop_secs=0.3,  # Reduced from default 0.8 - triggers faster after silence
                ),
            ),
            vad_audio_passthrough=True,
            audio_out_sample_rate=8000,
            serializer=serializer,
        )
    )

    await run_bot(transport, handle_sigint=False)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
