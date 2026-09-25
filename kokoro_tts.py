import asyncio
import numpy as np

from kokoro import KPipeline
from livekit import rtc


class KokoroTTS:
    def __init__(
        self,
        lang_code="a",
        voice="af_heart",
        sample_rate=24000,
    ):
        print("Loading Kokoro TTS...")

        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.sample_rate = sample_rate

        print("Kokoro TTS loaded.")

    async def synthesize(self, text: str):
        """
        Generate speech using Kokoro and return LiveKit AudioFrames.
        """

        if not text or not text.strip():
            return

        print(f"Kokoro TTS: {text}")

        def generate():
            audio_chunks = []

            generator = self.pipeline(
                text,
                voice=self.voice,
            )

            for _, _, audio in generator:
                audio_chunks.append(audio)

            if not audio_chunks:
                return None

            return np.concatenate(audio_chunks)

        audio = await asyncio.to_thread(generate)

        if audio is None:
            return

        # Kokoro returns float32 audio.
        # LiveKit AudioFrame expects 16-bit signed PCM.
        audio_int16 = np.clip(
            audio * 32767,
            -32768,
            32767,
        ).astype(np.int16)

        # Send audio in small frames.
        frame_size = self.sample_rate // 100  # 10 ms

        for start in range(0, len(audio_int16), frame_size):
            chunk = audio_int16[start:start + frame_size]

            if len(chunk) == 0:
                continue

            yield rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk),
            )