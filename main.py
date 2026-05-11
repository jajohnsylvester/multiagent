# --- Top of main.py: Critical SQLite Fix for Render ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel

# ADK Core Imports
from google.adk.agents import Agent, SequentialAgent
from google.adk.apps import App  # Unified orchestrator
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService 
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Research Service")

# --- 1. Agents ---
researcher = Agent(
    name="Researcher",
    instruction="We gather raw technical data and market facts.",
    model="gemini-2.0-flash"
)

synthesizer = Agent(
    name="Synthesizer",
    instruction="We transform raw research into a formal executive report.",
    model="gemini-2.0-flash"
)

# --- 2. Orchestration Components ---
root_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, synthesizer]
)

# SQLite with async driver for Render
if not os.path.exists("./data"):
    os.makedirs("./data")

db_url = "sqlite+aiosqlite:///data/sessions.db"
session_service = DatabaseSessionService(db_url)
APP_NAME = "ResearchLab"

# Create the unified App object (This fixes the 'Session not found' error)
adk_app = App(
    name=APP_NAME,
    root_agent=root_agent
)

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "online", "engine": "ADK App + DatabaseSession"}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"--- Starting Pipeline for: {request.topic} ---")
    final_text = ""
    
    # Identifiers
    USER_ID = "default_user"
    SESSION_ID = "research_session_fixed_01" 

    try:
        # Step 1: Explicitly ensure the session exists in DB
        try:
            await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
            logger.info("Existing session confirmed.")
        except Exception:
            logger.info("Initializing fresh session record...")
            await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

        # Step 2: Initialize Runner using the 'app' parameter
        # This creates the link between the agents and the session DB
        runner = Runner(
            app=adk_app,
            session_service=session_service
        )
        
        content = types.Content(
            role="user",
            parts=[types.Part(text=request.topic)]
        )

        # Step 3: Run the async stream
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

        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"PIPELINE CRASH: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
