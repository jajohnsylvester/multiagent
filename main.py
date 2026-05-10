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
        
        # Iterate through the async generator
        async for event in root_agent.run_async(request.topic):
            # 1. Check if the event has a 'text' attribute (Direct Content)
            if hasattr(event, 'text') and event.text:
                final_text = event.text
            # 2. Check if it's nested in content (Standard ADK Event structure)
            elif hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    final_text = event.content.parts[0].text
        
        if not final_text:
            # Fallback: some versions return the event itself as the result
            final_text = str(event) 

        # We return a clean dictionary. 
        # This prevents FastAPI/Pydantic from calling .model_copy() on a string.
        return {"status": "success", "report": str(final_text)}
        
    except Exception as e:
        print(f"Detailed Pipeline Error: {e}")
        # Return the error message as a string to avoid serialization issues
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Bind to Render's dynamic port
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
