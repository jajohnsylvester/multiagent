import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging to show in Render/Terminal console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from google.adk.agents import Agent
from google.adk.agents import SequentialAgent

app = FastAPI(title="Debuggable Research Agent")

# --- 1. Agents ---
researcher = Agent(
    name="Researcher",
    instruction="Collect raw data and facts on the topic.",
    model="gemini-2.0-flash"
)

synthesizer = Agent(
    name="Synthesizer",
    instruction="Create a structured report from the research provided.",
    model="gemini-2.0-flash"
)

# --- 2. Sequential Multi-Agent ---
root_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, synthesizer]
)

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "online"}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"--- Starting Pipeline for topic: {request.topic} ---")
    final_text = ""
    
    try:
        # Iterate through the async generator
        async for event in root_agent.run_async(request.topic):
            # Log the event type to see where the process is
            logger.info(f"Event Received: {type(event).__name__}")
            
            # Log event content if it's a model turn or result
            if hasattr(event, 'text') and event.text:
                logger.info(f"Text captured from {type(event).__name__}")
                final_text = event.text
            
            # Deep inspection of event content for debugging
            elif hasattr(event, 'content'):
                logger.info("Event has content attribute, extracting parts...")
                if hasattr(event.content, 'parts') and event.content.parts:
                    final_text = event.content.parts[0].text

        if not final_text:
            logger.warning("Pipeline finished but final_text is empty.")
            final_text = "Agents completed but returned no text."

        logger.info("--- Pipeline Completed Successfully ---")
        
        # We return a plain dict. 
        # This fixes the "'str' object has no attribute 'model_copy'" error.
        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        # This captures the traceback and logs it to Render
        logger.error(f"CRITICAL PIPELINE ERROR: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
