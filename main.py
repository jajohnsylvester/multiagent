import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Correct namespaced imports for ADK v1.33+
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService  # Stable Persistent Service
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Research Service")

# --- 1. Agents ---
researcher = Agent(
    name="Researcher",
    instruction="Collect raw technical data and financial facts.",
    model="gemini-2.0-flash"
)

synthesizer = Agent(
    name="Synthesizer",
    instruction="Transform raw data into a formal executive report.",
    model="gemini-2.0-flash"
)

root_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, synthesizer]
)

# --- 2. Persistent Database Session (SQLite) ---
# This replaces ChromaDBSessionService with the official stable equivalent
db_url = "sqlite:///sessions.db"
session_service = DatabaseSessionService(url=db_url)
APP_NAME = "ResearchLab"

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "online", "storage": "DatabaseSessionService (SQLite)"}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"--- Starting Pipeline for: {request.topic} ---")
    final_text = ""
    
    # We use a static user/session for now
    USER_ID = "default_user"
    SESSION_ID = "research_session_001"

    try:
        # Ensure session exists
        try:
            await session_service.get_session(APP_NAME, USER_ID, SESSION_ID)
        except Exception:
            logger.info(f"Initializing new session: {SESSION_ID}")
            await session_service.create_session(APP_NAME, USER_ID, SESSION_ID)

        runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=session_service
        )
        
        content = types.Content(
            role="user",
            parts=[types.Part(text=request.topic)]
        )

        async for event in runner.run_async(
            user_id=USER_ID, 
            session_id=SESSION_ID, 
            new_message=content
        ):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_text = part.text

        if not final_text:
            return {"status": "error", "message": "No output produced."}

        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"PIPELINE CRASH: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
