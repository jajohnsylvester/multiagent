import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Correct Namespaced Imports for Google ADK
from google.adk.agents import Agent
from google.adk.agents import SequentialAgent
from google.adk.runtime import Runtime

# Configure logging for Render console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Research Service")

# --- 1. Define Specialized Agents ---
# We use gemini-2.0-flash for high-speed research and synthesis
researcher = Agent(
    name="Researcher",
    instruction="We gather raw data, primary sources, and technical facts for the given topic.",
    model="gemini-2.0-flash"
)

synthesizer = Agent(
    name="Synthesizer",
    instruction="We take raw data from the Researcher and transform it into a formal executive report.",
    model="gemini-2.0-flash"
)

# --- 2. Create the Sequential Pipeline ---
root_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, synthesizer]
)

# Initialize the ADK Runtime (The shim that prevents the model_copy error)
runtime = Runtime()

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "ADK Web Enabled", "active": True}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"--- Starting Sequential Pipeline: {request.topic} ---")
    final_text = ""
    
    try:
        # We use runtime.stream to safely wrap the topic string into an InvocationContext
        # This prevents the AttributeError: 'str' object has no attribute 'model_copy'
        async for event in runtime.stream(root_agent, request.topic):
            event_type = type(event).__name__
            logger.info(f"Pipeline Event: {event_type}")
            
            # Extract content from the stream events
            if hasattr(event, 'text') and event.text:
                final_text = event.text
            elif hasattr(event, 'content') and hasattr(event.content, 'parts'):
                if event.content.parts:
                    final_text = event.content.parts[0].text

        if not final_text:
            logger.error("Pipeline reached end of stream without capturing text.")
            return {"status": "error", "message": "No final report generated."}

        logger.info("--- Pipeline Successful ---")
        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"PIPELINE CRASH: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Bind to Render's dynamic port or default to 10000
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
