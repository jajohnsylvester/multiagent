import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Correcting the imports to the flattened namespace
try:
    from adk import Agent, SequentialAgent, Runtime
    logger_msg = "Using flattened 'adk' namespace"
except ImportError:
    from google.adk.agents import Agent, SequentialAgent
    from google.adk.runtime import Runtime
    logger_msg = "Using 'google.adk' namespace"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(logger_msg)

app = FastAPI(title="Multi-Agent Research Service")

# --- 1. Agents ---
researcher = Agent(
    name="Researcher",
    instruction="We gather raw data and technical facts for the given topic.",
    model="gemini-2.0-flash"
)

synthesizer = Agent(
    name="Synthesizer",
    instruction="We transform raw research data into a formal executive report.",
    model="gemini-2.0-flash"
)

# --- 2. Pipeline ---
root_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, synthesizer]
)

runtime = Runtime()

class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health():
    return {"status": "Online", "mode": logger_msg}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"--- Starting Pipeline: {request.topic} ---")
    final_text = ""
    try:
        # Use runtime.stream to handle the context correctly
        async for event in runtime.stream(root_agent, request.topic):
            if hasattr(event, 'text') and event.text:
                final_text = event.text
            elif hasattr(event, 'content') and hasattr(event.content, 'parts'):
                if event.content.parts:
                    final_text = event.content.parts[0].text

        if not final_text:
            return {"status": "error", "message": "No output generated."}

        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"Pipeline Crash: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
