#
# Navana Bodhi Speech-to-Text service implementation for Pipecat.
#

"""Navana Bodhi speech-to-text service implementation."""

import asyncio
import audioop
import json
import uuid
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Navana STT, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")


class NavanaSTTService(STTService):
    """Navana Bodhi speech-to-text service.

    Provides real-time speech recognition using Navana Bodhi's WebSocket API.
    Supports Indian languages including Hindi, Bengali, Tamil, Telugu, Marathi,
    Kannada, Malayalam, Gujarati, Odia, and English (Indian).
    """

    def __init__(
        self,
        *,
        api_key: str,
        customer_id: str,
        model: str = "hi-general-v2-8khz",
        url: str = "wss://bodhi.navana.ai",
        sample_rate: Optional[int] = 8000,
        language: Language = Language.HI,
        parse_number: bool = True,
        exclude_partial: bool = False,
        **kwargs,
    ):
        """Initialize the Navana Bodhi STT service.

        Args:
            api_key: Navana Bodhi API key for authentication.
            customer_id: Navana Bodhi customer ID for authentication.
            model: Navana model name (e.g., "hi-general-v2-8khz").
            url: Navana Bodhi WebSocket URL.
            sample_rate: Audio sample rate. Defaults to 8000 for telephony.
            language: Language for speech recognition.
            parse_number: Convert spoken numerals to digits.
            exclude_partial: If True, only return final transcripts (no interim).
            **kwargs: Additional arguments passed to the parent STTService.
        """
        self._navana_sample_rate = sample_rate or 8000
        # Don't pass sample_rate to base class — let it use the transport's
        # native rate. We'll resample to _navana_sample_rate before sending.
        super().__init__(**kwargs)

        self._api_key = api_key
        self._customer_id = customer_id
        self._url = url
        self._model = model
        self._parse_number = parse_number
        self._exclude_partial = exclude_partial

        self._settings = {
            "language": language,
            "model": model,
        }

        self.set_model_name(model)

        self._websocket = None
        self._receive_task = None
        self._connected = False

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        """Set the Navana model and reconnect.

        Args:
            model: The Navana model name to use.
        """
        await super().set_model(model)
        logger.info(f"Switching STT model to: [{model}]")
        self._model = model
        self._settings["model"] = model
        await self._disconnect()
        await self._connect()

    async def set_language(self, language: Language):
        """Set the recognition language.

        Args:
            language: The language to use for speech recognition.
        """
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language"] = language

    async def start(self, frame: StartFrame):
        """Start the Navana STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        logger.info(f"{self}: Starting — pipeline sample_rate={self.sample_rate}Hz, "
                     f"navana target={self._navana_sample_rate}Hz, model={self._model}")
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Navana STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Navana STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Navana Bodhi for transcription.

        Resamples from the pipeline's sample rate to Navana's expected rate
        (typically 8kHz) if they differ.

        Args:
            audio: Raw PCM audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket receive task).
        """
        if self._websocket and self._connected:
            try:
                # Resample if pipeline rate differs from Navana's expected rate
                if self.sample_rate != self._navana_sample_rate:
                    audio, _ = audioop.ratecv(
                        audio, 2, 1,
                        self.sample_rate,
                        self._navana_sample_rate,
                        None,
                    )
                await self._websocket.send(audio)
            except Exception as e:
                logger.warning(f"{self} error sending audio: {e}")
                await self._reconnect()
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Navana-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            logger.debug(f"{self}: [VAD] User started speaking")
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.debug(f"{self}: [VAD] User stopped speaking")

    async def _connect(self):
        """Connect to Navana Bodhi WebSocket and send config."""
        logger.debug(f"Connecting to Navana Bodhi at {self._url}")

        try:
            extra_headers = {
                "x-api-key": self._api_key,
                "x-customer-id": self._customer_id,
            }

            self._websocket = await websockets.connect(
                self._url,
                additional_headers=extra_headers,
            )
            self._connected = True

            # Send config message
            config_message = {
                "config": {
                    "sample_rate": self._navana_sample_rate,
                    "transaction_id": str(uuid.uuid4()),
                    "model": self._model,
                    "parse_number": self._parse_number,
                    "exclude_partial": self._exclude_partial,
                }
            }
            await self._websocket.send(json.dumps(config_message))
            logger.debug(f"{self}: Sent config: {config_message}")

            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_messages())

            await self._call_event_handler("on_connected", self)
            logger.info(f"{self}: Connected to Navana Bodhi")

        except Exception as e:
            logger.error(f"{self}: Failed to connect to Navana Bodhi: {e}")
            self._connected = False
            await self._call_event_handler("on_connection_error", str(e))
            await self.push_error(error_msg=f"Navana connection error: {e}")

    async def _disconnect(self):
        """Disconnect from Navana Bodhi WebSocket."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._websocket and self._connected:
            try:
                # Send CloseStream signal
                await self._websocket.send(json.dumps({"type": "CloseStream"}))
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"{self}: Error during disconnect: {e}")
            finally:
                self._connected = False
                self._websocket = None
                await self._call_event_handler("on_disconnected", self)
                logger.debug(f"{self}: Disconnected from Navana Bodhi")

    async def _reconnect(self):
        """Reconnect to Navana Bodhi after an error."""
        logger.warning(f"{self}: Reconnecting to Navana Bodhi...")
        await self._disconnect()
        await self._connect()

    async def _receive_messages(self):
        """Receive and process messages from Navana Bodhi WebSocket."""
        try:
            async for message in self._websocket:
                if isinstance(message, bytes):
                    logger.warning(f"{self}: Received unexpected binary message, ignoring")
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"{self}: Received invalid JSON: {message}")
                    continue

                # Check for end-of-stream
                if data.get("eos", False):
                    logger.info(f"{self}: [EOS] End of stream received")
                    continue

                transcript = data.get("text", "").strip()
                msg_type = data.get("type", "")
                segment_id = data.get("segment_id", "")

                if not transcript:
                    continue

                is_final = msg_type == "complete"

                if is_final:
                    logger.info(f"{self}: [FINAL] segment={segment_id} \"{transcript}\"")
                    await self.stop_ttfb_metrics()
                    await self.push_frame(
                        TranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                        )
                    )
                    await self._handle_transcription(transcript, True)
                    await self.stop_processing_metrics()
                elif msg_type == "partial":
                    logger.debug(f"{self}: [PARTIAL] segment={segment_id} \"{transcript}\"")
                    await self.stop_ttfb_metrics()
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                        )
                    )

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"{self}: WebSocket connection closed: {e}")
            self._connected = False
            await self._call_event_handler("on_connection_error", str(e))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"{self}: Error receiving messages: {e}")
            self._connected = False
            await self._call_event_handler("on_connection_error", str(e))

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass
