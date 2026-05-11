import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Standard Google ADK imports for production environments
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner 

# Configure logging for Render console visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Research Service")

# --- 1. Define Agents ---
researcher = Agent(
    name="Researcher",
    instruction="We gather raw technical data and primary market facts for the given topic.",
    model="gemini-2.0-flash"
)

synthesizer = Agent(
    name="Synthesizer",
    instruction="We transform raw data from the Researcher into a formal executive report.",
    model="gemini-2.0-flash"
)

# --- 2. Sequential Orchestration ---
root_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, synthesizer]
)

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "online", "engine": "Google ADK Runner"}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"--- Starting Pipeline: {request.topic} ---")
    final_text = ""
    
    try:
        # The Runner prevents the 'str' has no attribute 'model_copy' error
        # by creating a valid InvocationContext internally.
        runner = Runner(agent=root_agent)
        
        async for event in runner.run_async(request.topic):
            # Extract content from the stream of agent events
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    # Capture the most recent text output in the sequence
                    part = event.content.parts[0]
                    if hasattr(part, 'text') and part.text:
                        final_text = part.text

        if not final_text:
            return {"status": "error", "message": "The agents failed to return a report."}

        logger.info("--- Pipeline Completed Successfully ---")
        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"PIPELINE CRASH: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Bind to Render's dynamic port (defaulting to 10000)
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
