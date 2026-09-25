from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, cli
from livekit.plugins import openai, silero

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


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM.with_ollama(
            model="llama3:8b",
            base_url="http://localhost:11434/v1",
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