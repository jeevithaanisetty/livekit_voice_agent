import asyncio
import uuid

import numpy as np
from faster_whisper import WhisperModel
from livekit.agents import stt, utils


class FasterWhisperSTT(stt.STT):
    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                diarization=False,
                offline_recognize=True,
            )
        )

        self._model_size = model_size
        self._language = language

        print(f"Loading faster-whisper model: {model_size}")

        self._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

        print("faster-whisper model loaded.")

    @property
    def model(self) -> str:
        return f"faster-whisper-{self._model_size}"

    @property
    def provider(self) -> str:
        return "faster-whisper-local"

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language=None,
        conn_options=None,
    ) -> stt.SpeechEvent:

        # AudioBuffer in LiveKit 1.8.3 can be either:
        # AudioFrame
        # or list[AudioFrame]
        if isinstance(buffer, list):
            frames = buffer
        else:
            frames = [buffer]

        # Safety check
        if not frames:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=str(uuid.uuid4()),
                alternatives=[],
            )

        # Combine raw PCM audio data from all frames
        audio_bytes = b"".join(
            bytes(frame.data)
            for frame in frames
        )

        # Get audio properties from the first frame
        sample_rate = frames[0].sample_rate
        num_channels = frames[0].num_channels

        # Convert PCM int16 bytes -> float32 audio
        audio = (
            np.frombuffer(
                audio_bytes,
                dtype=np.int16,
            )
            .astype(np.float32)
            / 32768.0
        )

        # Convert stereo/multi-channel audio to mono
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels)
            audio = audio.mean(axis=1)

        # faster-whisper works best with 16 kHz audio
        if sample_rate != 16000:
            old_length = len(audio)

            new_length = int(
                old_length * 16000 / sample_rate
            )

            if old_length > 0 and new_length > 0:
                old_positions = np.linspace(
                    0,
                    1,
                    old_length,
                )

                new_positions = np.linspace(
                    0,
                    1,
                    new_length,
                )

                audio = np.interp(
                    new_positions,
                    old_positions,
                    audio,
                ).astype(np.float32)

        # Use requested language if LiveKit provides one
        whisper_language = self._language

        if language:
            whisper_language = language

        # Run Whisper in a background thread
        segments, info = await asyncio.to_thread(
            self._transcribe,
            audio,
            whisper_language,
        )

        # Combine Whisper segments into one transcript
        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()

        print(f"Whisper transcript: {text}")

        # Return final transcript to LiveKit
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[
                stt.SpeechData(
                    language=whisper_language or "en",
                    text=text,
                    confidence=1.0,
                )
            ],
        )

    def _transcribe(
        self,
        audio,
        language,
    ):
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=False,
        )

        return list(segments), info

    async def aclose(self):
        pass