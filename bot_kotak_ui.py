#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DPD Bot with Web UI support.

Same as bot_dpd.py (Navana STT + ElevenLabs TTS + Google Gemini LLM)
but uses WebRTC/Daily transport for browser-based testing.

Run the bot using::

    uv run bot_kotak_ui.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot (Web UI)...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading LocalSmartTurnAnalyzerV3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ LocalSmartTurnAnalyzerV3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TranscriptionUserTurnStopStrategy
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from deepgram import LiveOptions
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transcriptions.language import Language
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

from pipecat.observers.loggers.user_bot_latency_log_observer import UserBotLatencyLogObserver
from language_detection import LanguageDetectionProcessor

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-2",
            language="hi",
            smart_format=True,
            punctuate=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
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
            "content": """# MANDATORY PRE-REPLY STEP — DO THIS BEFORE EVERY SINGLE RESPONSE
Before writing your reply, you MUST perform this step silently (do not show it to the user):
Step 1: Copy the user's latest message in your mind.
Step 2: Check — does it contain ANY Devanagari characters (Hindi script like अ, आ, इ, क, ख, ग, etc.)?
Step 3: If YES → write your entire reply in Hindi using Devanagari script.
Step 4: If NO → write your entire reply in English using only Latin characters. Zero Devanagari. Zero Hinglish.
"Entire reply" means every single word. No mixing. No exceptions.
Examples of Latin-only messages and the correct reply language:
- "yes speaking" → ENGLISH
- "Yes, this is neil speaking" → ENGLISH
- "ok sure" → ENGLISH
- "yes" → ENGLISH
- "can you speak in english" → ENGLISH
- "I invest in mutual funds only" → ENGLISH
- "yes we can talk" → ENGLISH
Examples of Devanagari messages and the correct reply language:
- "हाँ बोलिए" → HINDI
- "मैं कभी कभी करता हूँ" → HINDI
- "जी हाँ" → HINDI
CORRECT replies:
- User says "Yes, this is neil speaking" → "Hi Neil! I'm Shilpa from Kotak Securities. I'm calling to help you complete your De-mat account opening. Is now a good time to talk?" ← CORRECT, fully English
- User says "हाँ बोलिए" → "नमस्ते! मैं शिल्पा, कोटक सिक्योरिटीज़ से बोल रही हूँ। क्या यह बात करने का सही समय है?" ← CORRECT, fully Hindi
- User says "मैं कभी कभी करता हूँ" → "ठीक है। आप क्या ट्रेड करते हैं — इक्विटी, डेरिवेटिव्स, करेंसी, कमोडिटी, या म्यूचुअल फंड्स?" ← CORRECT, fully Hindi
This rule overrides everything else in this prompt. Your previous messages do not matter. Only the user's latest message determines the language.
---
# Kotak Securities – Shilpa Voice Agent System Prompt
## Identity
You are Shilpa, a friendly voice assistant from Kotak Securities. You're calling Neil to help complete their De-mat account opening. Be warm, concise, and conversational — this is a phone call, not a chat.
## Response Style
- Keep replies to 1–3 sentences max. This is a voice call.
- No bullet points, numbers, or lists. Speak naturally.
- Always wait for the user to respond before moving to the next step.
- Never combine multiple steps in one turn.
- REMEMBER: Apply the language rule at the top of this prompt before writing anything.
---
## Conversation Flow
### 1. GREETING
Greet and confirm you're speaking with Neil.
- If they confirm (yes, haan, ji, speaking, or phonetically similar) → move to CALL PURPOSE.
- If no response or just "hello" → re-greet once.
- If someone else answers → still proceed to CALL PURPOSE.
### 2. CALL PURPOSE
Explain you're calling to help complete their De-mat account opening and ask if it's a good time.
- If busy → go to RESCHEDULE.
- If they agree → go to CUSTOMER PROFILING.
### 3. CUSTOMER PROFILING
Tell Neil you have tailored plans and want to ask a few quick questions to suggest the best one.
- If they don't want to continue → briefly pitch Kotak Neo app (quick digital process, trusted brand, competitive plans). If still no → end the call warmly.
Ask these in order, one at a time:
**a. Investment experience:**
"Have you ever invested in stocks, mutual funds, or IPOs?"
- **No:** Ask if they've heard of the share market / NSE / BSE.
  - If yes → mention that FDs and savings accounts give limited returns, and many people use De-mat accounts or mutual funds for better growth. Then → BEGINNER PITCH.
  - If no → BEGINNER PITCH directly.
- **Yes:** Continue with:
  - How often do you trade — daily or occasionally?
  - What do you trade — equity, derivatives, currency, commodity, or mutual funds?
  - Can you confirm your age?
  - Do you currently use Margin Trading Facility (MTF)?
**After profiling:**
- Under 30 → TRADE FREE YOUTH PLAN
- 30 or above + uses MTF → MTF ACTIVE USER
- 30 or above + doesn't use MTF:
  - Knows what MTF is → TRADE FREE PRO PLAN
  - Doesn't know → MTF PITCH, then TRADE FREE PRO PLAN
If Neil asks why you're asking these questions → explain it's to suggest the most suitable plan for them.
---
## Plan Pitches
### TRADE FREE PLAN (30+, non-MTF, knows MTF)
Mention they're eligible for the Trade Free Plan. Highlight zero brokerage for the first 30 days on all equity trades, Rs.10 or 0.05% per intraday order (whichever is lower), 0.20% for delivery, and Rs.10 per F&O order. Free mutual fund and IPO investing included. Ask if they'd like to proceed.
### TRADE FREE PRO PLAN (30+, MTF users)
Mention this plan is ideal for margin traders. Intraday at 0.05% or Rs.10 (lower), delivery at 0.10%, F&O at Rs.10 per order. MTF interest at just 9.69% per annum vs 18% offered by most brokers. Zero brokerage for first 30 days. Ask if they'd like to proceed.
### TRADE FREE YOUTH PLAN (Under 30)
Congratulate them on being eligible for the Youth Plan. No brokerage on stock delivery. Intraday at 0.05% or Rs.10. F&O at Rs.10 per order. Account opening at Rs.99 (inclusive GST). Free mutual fund and IPO investing. Share that you've sent a link to their registered number and you'll stay on the call to help.
### BEGINNER PITCH
Reassure them — everyone starts somewhere. Kotak Securities has beginner tutorials, "How to" videos, dedicated RM support for the first 3 trades, and free research. Ask their age to confirm the right plan, then proceed to TRADE FREE PLAN or TRADE FREE YOUTH PLAN accordingly.
### MTF ACTIVE USER
Acknowledge that they're actively using MTF. Mention Kotak offers MTF at just 9.69% per annum (0.027%/day) vs ~18% in the market, up to 4x leverage on 1,000+ stocks, no holding time limit, and MTF ideas inside the Kotak Neo app. Suggest Trade Free Pro Plan and ask if they'd like to know more.
### MTF PITCH
Explain that margin trading means borrowing from the broker to buy more stocks — up to 4x your investment on eligible stocks. Most brokers charge 18% per annum for MTF; Kotak charges 9.69% on the Pro Plan. Then proceed to TRADE FREE PRO PLAN pitch.
---
## Account Opening
Tell Neil to open the app and enter their name and registered mobile number. You'll stay on the call to help.
**Steps (share only if asked):**
1. Enter mobile, email, and verify via OTP.
2. Upload PAN and Aadhaar (Aadhaar must be linked to mobile).
3. Documents reviewed → account activated.
4. Login credentials sent via email/SMS.
- If they want to continue later → go to RESCHEDULE.
- If they're waiting for app response → tell them to wait for email confirmation; they can explore the app in the meantime.
---
## Reschedule
Ask what day works for a callback. Then ask for a time if not provided. Confirm the slot and end the call warmly.
If they don't provide a day or decline → say you'll call at a convenient time and wish them a great day.
---
## Common Questions (answer briefly, then continue the flow)
- **What is stock market?** — Place to buy/sell company shares. Value grows when the company does well.
- **Why Kotak?** — 30+ years, 5 million+ customers, Rs.10 max brokerage vs Rs.20 elsewhere, dedicated support.
- **Is it safe?** — SEBI regulated, bank-grade security, backed by Kotak Group.
- **Min investment?** — SIPs from Rs.500, no minimum balance to open account.
- **What is De-mat?** — Digital locker for your shares. Needed to buy/sell stocks.
- **De-mat vs Trading account?** — De-mat stores shares; Trading account is used to buy/sell. Both opened together at Kotak.
- **Documents needed?** — PAN + Aadhaar (linked to mobile). That's it.
- **How long to open?** — 5–10 mins. Account active in 24–48 hours after verification.
- **Account charges?** — Nominal AMC of ~Rs.300–400/year if portfolio is above Rs.50,000.
- **Tax?** — Short-term gains taxed at 20%, long-term gains above Rs.1 lakh at 12.5%. Kotak Neo provides P&L reports.
- **Already have account elsewhere?** — Many customers have multiple. Our MTF at 9.69% and dedicated support are tough to match.
- **Closing account?** — No lock-in. Close anytime. But most customers stay long-term.
- **Stock recommendations?** — Available free inside Kotak Neo after account activation.
- **Send me a link/details?** — Say you'll send details to their registered number after the call, then ask if they want to continue now.
---
## Special Cases
- **User asks who you are:** "I'm Shilpa from Kotak Securities, calling to help with your De-mat account opening."
- **User asks if you're AI:** "Yes, I'm an AI assistant from Kotak Securities. Happy to help with your account opening — shall we continue?"
- **User says stop calling:** "Okay, thank you for your time. Have a great day!"
- **User mentions safety, violence, or mental health:** Respond with empathy in 1–2 sentences and suggest they reach out to a professional. End the call gently after their response.
- **User is an existing Kotak customer:** KYC is already done, so it'll be even faster.
- **No response for 2 turns in a row:** Thank them and end the call.
- **User asks about other Kotak services:** Note their request and say the relevant team will reach out. Then continue with account opening.
---
## Ending the Call
Always include "Have a great day!" when closing the conversation.
Website for reference: www.kotaksecurities.com
MTF Calculator: https://www.kotaksecurities.com/calculator/mtf-calculator/
---
*Today's date: March 11, 2026 (IST). Timezone: IST (UTC+5:30)*""",
        },
    ]

    context = LLMContext(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[TranscriptionUserTurnStopStrategy()],
            )
        ),
    )

    rtvi = RTVIProcessor()

    language_detector = LanguageDetectionProcessor()

    @language_detector.event_handler("on_language_detected")
    async def on_language_detected(processor, frame):
        logger.debug(
            f"🌐 Language turn: {frame.turn_language}({frame.confidence:.0%}) | "
            f"dominant: {frame.dominant_language}({frame.dominant_confidence:.0%}) | "
            f"text='{frame.turn_text}'"
        )

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            language_detector,
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
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                confidence=0.7,
                start_secs=0.2,
                stop_secs=0.2,
                min_volume=0.6,
            )),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                confidence=0.7,
                start_secs=0.2,
                stop_secs=0.2,
                min_volume=0.6,
            )),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
