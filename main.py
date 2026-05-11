import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Correct namespaced imports for ADK v1.33+
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService  # Added for state management
from google.genai import types  # Added for proper message formatting

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Research Service")

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

# --- 3. Session Initialization (The Fix) ---
# We initialize the session service once
session_service = InMemorySessionService()
APP_NAME = "ResearchLab"

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "online"}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"--- Starting Pipeline for: {request.topic} ---")
    final_text = ""
    
    try:
        # 1. Initialize the Runner with the required session service
        runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=session_service
        )
        
        # 2. Format the input as a GenAI Content object (best practice for Runners)
        content = types.Content(
            role="user",
            parts=[types.Part(text=request.topic)]
        )

        # 3. Execute with user and session identifiers
        # We use a static user_id since we are not tracking individual users yet
        async for event in runner.run_async(
            user_id="default_user", 
            session_id="research_session", 
            new_message=content
        ):
            # Extract content from event parts
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_text = part.text

        if not final_text:
            raise Exception("No final response text captured from the stream.")

        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"PIPELINE CRASH: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
