import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    room_io,
)
from livekit.plugins import google

load_dotenv(".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")


class VoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                            You are a friendly AI voice assistant.
                            Your job is to have a natural conversation with the user.
                            Listen carefully to what the user says.
                            Answer their questions clearly and accurately.
                            Keep your spoken responses concise.
                            Do not use markdown.
                            Do not behave like a text chatbot.
                            Speak naturally like a helpful human assistant.
                            If you do not know something, say that you do not know.
                            """
                         )
server = AgentServer()

@server.rtc_session(agent_name="voice-assistant")
async def entrypoint(ctx: JobContext):
    logger.info(
        "Agent starting for room: %s",
        ctx.room.name
    )
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            voice="Puck",
            temperature=0.7,
        ),
    )

    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(),
        room_options=room_io.RoomOptions(
            audio_input=True,
            audio_output=True,
        ),
    )

    await ctx.connect()
    await session.generate_reply(
        instructions="""
                        Greet the user.

                        Say that you are a realtime AI voice assistant.

                        Ask the user how you can help them.
                    """
    )

if __name__ == "__main__":
    agents.cli.run_app(server)