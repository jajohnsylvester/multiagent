import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Correct namespaced imports for google-adk
from google.adk.agents import Agent
from google.adk.tools import AgentTool
from google.adk.agents import SequentialAgent

app = FastAPI(title="Research Agent Web Service")

# --- 1. Define Specialized Agents ---
researcher = Agent(
    name="Researcher",
    instruction="We gather raw technical data and market facts.",
    model="gemini-2.0-flash"
)

synthesizer = Agent(
    name="Synthesizer",
    instruction="We transform raw data into professional executive reports.",
    model="gemini-2.0-flash"
)

# --- 2. Sequential Multi-Agent Orchestration ---
# This 'root_agent' is what ADK Web and CLI tools look for by default
root_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, synthesizer]
)

# --- 3. FastAPI Endpoints for Render ---
class ResearchRequest(BaseModel):
    topic: str

@app.get("/")
def health_check():
    return {"status": "ADK Web Enabled", "active": True}

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    try:
        final_text = ""
        
        # We use 'async for' to pull events from the sequential agent stream
        async for event in root_agent.run_async(request.topic):
            # Check if this specific event is the final text response
            if hasattr(event, 'text') and event.text:
                final_text = event.text
            elif hasattr(event, 'is_final_response') and event.is_final_response():
                # Some versions of ADK provide a content object in the event
                final_text = event.content.parts[0].text

        if not final_text:
            raise HTTPException(status_code=500, detail="No final response received from agents.")

        return {"status": "success", "report": final_text}
        
    except Exception as e:
        print(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Bind to Render's dynamic port
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
