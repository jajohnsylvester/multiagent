# --- Top of main.py: SQLite Fix for Render ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel

# ADK Core Imports
from google.adk.agents import Agent, SequentialAgent
from google.adk.runtime import Runtime
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

# --- 2. Pipeline ---
root_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, synthesizer]
)

# --- 3. Persistent Storage (SQLite Async) ---
if not os.path.exists("./data"):
    os.makedirs("./data")

db_url = "sqlite+aiosqlite:///data/sessions.db"
session_service = DatabaseSessionService(db_url)

# Initialize the Runtime with the session service
# The Runtime handles the "Session Not Found" logic automatically
runtime = Runtime(session_service=session_service)

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "online", "engine": "ADK Runtime + Persistent DB"}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"--- Starting Pipeline: {request.topic} ---")
    final_text = ""
    
    # We use a unique session ID per request or a shared one
    # Note: Runtime.stream will create this session if it doesn't exist
    SESSION_ID = "main_research_session" 

    try:
        # Runtime.stream is the safest way to execute on Render.
        # It manages the 'InvocationContext' and session lifecycle internally.
        async for event in runtime.stream(
            root_agent, 
            request.topic,
            session_id=SESSION_ID
        ):
            # Extracting content from the event stream
            if hasattr(event, 'text') and event.text:
                final_text = event.text
            elif hasattr(event, 'content') and hasattr(event.content, 'parts'):
                if event.content.parts:
                    final_text = event.content.parts[0].text

        if not final_text:
            return {"status": "error", "message": "The pipeline did not produce output."}

        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"PIPELINE CRASH: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
