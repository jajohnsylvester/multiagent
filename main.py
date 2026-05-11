# --- Top of main.py: SQLite Fix for Render ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel

# --- ROBUST ADK IMPORT SHIM ---
try:
    # Try the most common flattened namespace first
    from adk import Agent, SequentialAgent, Runtime
    from adk.sessions import DatabaseSessionService
    logger_msg = "Successfully imported from 'adk' namespace"
except ImportError:
    try:
        # Try the nested google.adk namespace
        from google.adk.agents import Agent, SequentialAgent
        from google.adk.runtime import Runtime
        from google.adk.sessions import DatabaseSessionService
        logger_msg = "Successfully imported from 'google.adk' namespace"
    except ImportError as e:
        logger_msg = f"CRITICAL IMPORT ERROR: {str(e)}"
        # Fallbacks for specific agents if still failing
        Agent = None
        Runtime = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(logger_msg)

app = FastAPI(title="Multi-Agent Research Service")

# Only initialize if imports succeeded
if Agent and Runtime:
    # --- 1. Agents ---
    researcher = Agent(
        name="Researcher",
        instruction="Gather raw technical data and market facts.",
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

    # --- 2. Persistence ---
    if not os.path.exists("./data"):
        os.makedirs("./data")

    db_url = "sqlite+aiosqlite:///data/sessions.db"
    session_service = DatabaseSessionService(db_url)
    runtime = Runtime(session_service=session_service)
else:
    logger.error("Service cannot start due to missing ADK components.")

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "online", "diagnostic": logger_msg}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    if not Runtime:
        return {"status": "error", "message": "ADK Runtime not loaded."}

    logger.info(f"--- Starting Pipeline: {request.topic} ---")
    final_text = ""
    
    try:
        # Use runtime.stream for the most stable execution flow
        async for event in runtime.stream(
            root_agent, 
            request.topic,
            session_id="main_research_session"
        ):
            if hasattr(event, 'text') and event.text:
                final_text = event.text
            elif hasattr(event, 'content') and hasattr(event.content, 'parts'):
                if event.content.parts:
                    final_text = event.content.parts[0].text

        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"PIPELINE CRASH: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
