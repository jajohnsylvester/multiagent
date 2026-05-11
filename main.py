# --- Top of main.py: SQLite Fix for ChromaDB on Render ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import ChromaDBSessionService # Use ChromaDB for persistence
from google.genai import types

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

# --- 2. Persistent Storage (ChromaDB) ---
db_path = "./adk_db"
if not os.path.exists(db_path):
    os.makedirs(db_path)

session_service = ChromaDBSessionService(path=db_path)
APP_NAME = "ResearchLab"

class ResearchRequest(BaseModel):
    topic: str

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"--- Starting Pipeline for: {request.topic} ---")
    final_text = ""
    
    # Configuration
    USER_ID = "default_user"
    SESSION_ID = "research_session_001" # Unique ID

    try:
        # --- THE FIX: Ensure session exists before running ---
        try:
            await session_service.get_session(APP_NAME, USER_ID, SESSION_ID)
            logger.info(f"Existing session found: {SESSION_ID}")
        except Exception:
            logger.info(f"Creating new session: {SESSION_ID}")
            await session_service.create_session(
                app_name=APP_NAME, 
                user_id=USER_ID, 
                session_id=SESSION_ID
            )

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

        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"PIPELINE CRASH: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
