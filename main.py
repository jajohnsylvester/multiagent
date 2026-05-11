# --- Top of main.py: SQLite Async Fix ---
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

# Configure logging first to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- DYNAMIC ADK RESOLVER ---
Agent, SequentialAgent, Runtime, DatabaseSessionService, Runner = None, None, None, None, None

def resolve_adk():
    global Agent, SequentialAgent, Runtime, DatabaseSessionService, Runner
    # List of possible paths for core components
    import_paths = [
        ('google.adk.agents', ['Agent', 'SequentialAgent']),
        ('google.adk.runtime', ['Runtime']),
        ('google.adk.runners', ['Runner']),
        ('google.adk.sessions', ['DatabaseSessionService']),
        ('adk', ['Agent', 'SequentialAgent', 'Runtime', 'Runner']),
    ]
    
    for path, objects in import_paths:
        try:
            module = __import__(path, fromlist=objects)
            for obj in objects:
                if hasattr(module, obj):
                    globals()[obj] = getattr(module, obj)
            logger.info(f"Successfully imported components from {path}")
        except ImportError:
            continue

resolve_adk()

# Verify imports
if not (Agent and (Runtime or Runner)):
    logger.error("FATAL: Could not resolve ADK components. Check requirements.txt.")
# -----------------------------

app = FastAPI(title="Multi-Agent Research Service")

# --- 1. Agents ---
if Agent:
    researcher = Agent(
        name="Researcher",
        instruction="Gather raw technical data and market facts.",
        model="gemini-2.0-flash"
    )
    synthesizer = Agent(
        name="Synthesizer",
        instruction="Transform raw research into a formal executive report.",
        model="gemini-2.0-flash"
    )
    root_agent = SequentialAgent(
        name="ResearchPipeline",
        sub_agents=[researcher, synthesizer]
    )
    
    # Setup Database Session
    if not os.path.exists("./data"): os.makedirs("./data")
    db_url = "sqlite+aiosqlite:///data/sessions.db"
    session_service = DatabaseSessionService(db_url) if DatabaseSessionService else None

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "online", "agent_loaded": Agent is not None, "runner_ready": (Runtime or Runner) is not None}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    if not (Agent and (Runtime or Runner)):
        return {"status": "error", "message": "ADK Components not loaded properly."}

    logger.info(f"--- Starting Pipeline: {request.topic} ---")
    final_text = ""
    
    try:
        # We prefer Runtime for stability, fall back to Runner if needed
        if Runtime:
            runtime_engine = Runtime(session_service=session_service)
            async for event in runtime_engine.stream(root_agent, request.topic, session_id="main_session"):
                if hasattr(event, 'text') and event.text: final_text = event.text
        elif Runner:
            runner_engine = Runner(agent=root_agent, session_service=session_service)
            async for event in runner_engine.run_async(request.topic):
                if hasattr(event, 'text') and event.text: final_text = event.text

        return {"status": "success", "report": str(final_text)}
    except Exception as e:
        logger.error(f"PIPELINE CRASH: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
