import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ADK Imports
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import ChromaDBSessionService  # The Persistent Fix
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Research Service (Persistent)")

# --- 1. Agents ---
researcher = Agent(
    name="Researcher",
    instruction="Collect raw technical data and financial facts for the topic.",
    model="gemini-2.0-flash"
)

synthesizer = Agent(
    name="Synthesizer",
    instruction="Transform raw data from the Researcher into a formal executive report.",
    model="gemini-2.0-flash"
)

# --- 2. Sequential Multi-Agent ---
root_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, synthesizer]
)

# --- 3. ChromaDB Initialization ---
# We store the database in a local folder called 'adk_db'
# On Render, this will persist across deployments if using a Disk, 
# or reset on every restart on the Free Tier.
db_path = "./adk_db"
if not os.path.exists(db_path):
    os.makedirs(db_path)

session_service = ChromaDBSessionService(path=db_path)
APP_NAME = "ResearchLab_Persistent"

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "online", "storage": "ChromaDB"}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"--- Starting Persistent Pipeline: {request.topic} ---")
    final_text = ""
    
    try:
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
            user_id="default_user", 
            session_id="shared_research_history", # All queries saved here
            new_message=content
        ):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_text = part.text

        if not final_text:
            raise Exception("Pipeline execution failed to produce text.")

        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"PIPELINE CRASH: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
