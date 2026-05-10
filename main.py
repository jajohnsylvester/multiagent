import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Correcting the imports for the ADK and Google Gen AI
from adk import Agent, AgentTool
from adk.agents import SequentialAgent

app = FastAPI()

# --- 1. Define Specialized Agents ---
researcher = Agent(
    name="Researcher",
    instruction="We gather raw data and primary sources for the given topic.",
    model="gemini-2.0-flash"
)

synthesizer = Agent(
    name="Synthesizer",
    instruction="We take the Researcher's output and create a formal report with executive summaries.",
    model="gemini-2.0-flash"
)

# --- 2. Create the Sequential Pipeline ---
research_pipeline = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, synthesizer]
)

# --- 3. API Endpoints ---
class ResearchRequest(BaseModel):
    topic: str

@app.post("/research")
async def run_pipeline(request: ResearchRequest):
    try:
        # Running the sequential logic
        response = await research_pipeline.run_async(input=request.topic)
        return {"status": "success", "report": response.text}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ADK Web Enabled", "active": True}

if __name__ == "__main__":
    import uvicorn
    # Render provides the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
