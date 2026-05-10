import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Standard imports for Google ADK
try:
    from google.adk.agents import Agent
    from google.adk.agents.sequential_agent import SequentialAgent
    from google.adk.runtime import Runtime
    logger_msg = "Successfully imported google.adk"
except ImportError as e:
    # If this fails, we will log exactly what is missing
    logger_msg = f"Import failed: {str(e)}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(logger_msg)

app = FastAPI(title="Multi-Agent Research Service")

# --- 1. Agents ---
researcher = Agent(
    name="Researcher",
    instruction="Gather raw data and technical facts for the topic.",
    model="gemini-2.0-flash"
)

synthesizer = Agent(
    name="Synthesizer",
    instruction="Transform raw research into a professional report.",
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
    return {"status": "Online", "diagnostic": logger_msg}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    logger.info(f"Starting Pipeline: {request.topic}")
    final_text = ""
    try:
        # Runtime.stream is the safest way to execute to avoid context errors
        async for event in runtime.stream(root_agent, request.topic):
            if hasattr(event, 'text') and event.text:
                final_text = event.text
            elif hasattr(event, 'content') and hasattr(event.content, 'parts'):
                if event.content.parts:
                    final_text = event.content.parts[0].text

        if not final_text:
            return {"status": "error", "message": "Pipeline did not produce text."}

        return {"status": "success", "report": str(final_text)}

    except Exception as e:
        logger.error(f"Execution Error: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
