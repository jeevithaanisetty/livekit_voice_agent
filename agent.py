from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, cli
from livekit.plugins import openai, silero
from livekit.agents.stt import StreamAdapter
from faster_whisper_stt import FasterWhisperSTT
from livekit import rtc
from kokoro_tts import KokoroTTS

load_dotenv(".env")

class VoiceAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a helpful voice assistant.
            Keep your answers short and conversational.
            Speak naturally and clearly.
            """
        )

        self.kokoro = KokoroTTS(
            lang_code="a",
            voice="af_heart",
        )

    async def tts_node(self, text, model_settings):
        buffer = ""

        async for text_chunk in text:
            buffer += text_chunk

            # Generate audio when we have a complete sentence.
            while any(punctuation in buffer for punctuation in [".", "!", "?", "\n"]):
                split_position = len(buffer)

                for punctuation in [".", "!", "?", "\n"]:
                    position = buffer.find(punctuation)

                    if position != -1:
                        split_position = min(
                        split_position,
                        position + 1,
                        )

                sentence = buffer[:split_position].strip()
                buffer = buffer[split_position:]

                if sentence:
                    print(f"Kokoro TTS sentence: {sentence}")

                    async for frame in self.kokoro.synthesize(sentence):
                        yield frame

        # Generate any remaining text.
        remaining = buffer.strip()

        if remaining:
            print(f"Kokoro TTS sentence: {remaining}")

            async for frame in self.kokoro.synthesize(remaining):
                yield frame


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx):
    whisper_stt = FasterWhisperSTT(
        model_size="base",
        language="en",
    )

    stt = StreamAdapter(
        stt=whisper_stt,
        vad=silero.VAD.load(),
    )

    session = AgentSession(
        stt=stt,
        llm=openai.LLM(
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=30.0,
        ),
    )
    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(),
    )

    await session.generate_reply(
        instructions="Greet the user and briefly introduce yourself."
    )


if __name__ == "__main__":
    cli.run_app(server)