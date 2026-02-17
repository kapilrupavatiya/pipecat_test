#!/usr/bin/env python3
"""Standalone test for Navana Bodhi STT WebSocket connection.

Tests:
1. WebSocket connection with auth headers
2. Config message exchange
3. Streaming a WAV file and receiving transcription concurrently

Usage:
    python test_navana.py                          # Connection test (sends silence)
    python test_navana.py /path/to/audio.wav       # Test with a real audio file

Requires BODHI_API_KEY and BODHI_CUSTOMER_ID in .env or environment.
"""

import asyncio
import json
import os
import struct
import sys
import uuid

from dotenv import load_dotenv

load_dotenv(override=True)

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)


URL = "wss://bodhi.navana.ai"
SAMPLE_RATE = 8000
MODEL = "hi-general-v2-8khz"


def generate_silence(duration_s: float = 2.0) -> bytes:
    num_samples = int(SAMPLE_RATE * duration_s)
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def read_wav_as_pcm(filepath: str) -> tuple[bytes, int]:
    import wave
    with wave.open(filepath, "rb") as wf:
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
        channels = wf.getnchannels()
        print(f"  WAV: {sr}Hz, {channels}ch, {wf.getsampwidth() * 8}-bit, "
              f"{wf.getnframes() / sr:.1f}s")
        if sr != SAMPLE_RATE:
            print(f"  WARNING: WAV sample rate is {sr}Hz, Bodhi expects {SAMPLE_RATE}Hz")
        return pcm, sr


async def send_audio(ws, pcm_data: bytes):
    """Send audio in 1-second chunks (fast, not real-time)."""
    chunk_size = SAMPLE_RATE * 2  # 1 second of 16-bit mono
    total_duration = len(pcm_data) / (SAMPLE_RATE * 2)
    bytes_sent = 0

    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i : i + chunk_size]
        await ws.send(chunk)
        bytes_sent += len(chunk)
        elapsed_s = bytes_sent / (SAMPLE_RATE * 2)
        print(f"\r  [SEND] {elapsed_s:.1f}s / {total_duration:.1f}s", end="", flush=True)
        await asyncio.sleep(0.005)  # tiny yield for receive task

    print(f"\n  [SEND] Done: {len(pcm_data)} bytes ({total_duration:.1f}s)")

    # Signal end of stream
    await ws.send(json.dumps({"type": "CloseStream"}))
    print("  [SEND] Sent CloseStream")


async def receive_messages(ws, received: list, done_event: asyncio.Event):
    """Receive transcription messages until EOS or timeout."""
    try:
        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=10.0)
            if isinstance(message, bytes):
                continue

            data = json.loads(message)
            msg_type = data.get("type", "?")
            text = data.get("text", "")
            eos = data.get("eos", False)

            received.append(data)

            if text:
                label = "FINAL" if msg_type == "complete" else "partial"
                print(f"  [{label}] \"{text}\"")

            if eos:
                print("  [EOS] End of stream received")
                break
    except asyncio.TimeoutError:
        print("  [RECV] Timeout — no more messages")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"  [RECV] Connection closed: {e}")
    except asyncio.CancelledError:
        pass
    finally:
        done_event.set()


async def test_connection():
    api_key = os.getenv("BODHI_API_KEY")
    customer_id = os.getenv("BODHI_CUSTOMER_ID")

    if not api_key or not customer_id:
        print("ERROR: Set BODHI_API_KEY and BODHI_CUSTOMER_ID in .env or environment")
        sys.exit(1)

    # Step 1: Connect
    print(f"[1/3] Connecting to {URL}...")
    headers = {
        "x-api-key": api_key,
        "x-customer-id": customer_id,
    }
    try:
        ws = await websockets.connect(URL, additional_headers=headers)
        print("  OK: Connected")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    # Step 2: Send config
    print("[2/3] Sending config...")
    config = {
        "config": {
            "sample_rate": SAMPLE_RATE,
            "transaction_id": str(uuid.uuid4()),
            "model": MODEL,
            "parse_number": True,
            "exclude_partial": False,
        }
    }
    await ws.send(json.dumps(config))
    print(f"  OK: model={MODEL}, sample_rate={SAMPLE_RATE}")

    # Step 3: Stream audio + receive concurrently
    print("[3/3] Streaming audio + receiving transcriptions...")
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None

    if audio_file:
        if not os.path.exists(audio_file):
            print(f"  FAIL: File not found: {audio_file}")
            await ws.close()
            sys.exit(1)
        pcm_data, _ = read_wav_as_pcm(audio_file)
    else:
        print("  No audio file — sending 2s silence (connection test)")
        pcm_data = generate_silence(2.0)

    received = []
    done_event = asyncio.Event()

    # Run send and receive concurrently
    recv_task = asyncio.create_task(receive_messages(ws, received, done_event))
    await send_audio(ws, pcm_data)

    # Wait for receive to finish (EOS or timeout)
    await done_event.wait()
    recv_task.cancel()
    try:
        await recv_task
    except asyncio.CancelledError:
        pass

    # Clean up
    try:
        await ws.close()
    except Exception:
        pass

    # Summary
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    finals = [r for r in received if r.get("type") == "complete"]
    partials = [r for r in received if r.get("type") == "partial"]
    print(f"  Total messages: {len(received)}")
    print(f"  Final transcripts: {len(finals)}")
    print(f"  Partial transcripts: {len(partials)}")
    print()
    for i, f in enumerate(finals, 1):
        print(f"  Final #{i}: \"{f.get('text', '')}\"")

    if received:
        print(f"\n  Raw last response:")
        print(f"  {json.dumps(received[-1], indent=2, ensure_ascii=False)}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(test_connection())
