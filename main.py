import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.adk import Agent, AgentTool
from google.adk.agents import SequentialAgent

app = FastAPI()

# --- 1. Define Specialized Agents ---
# The Researcher (Step 1)
researcher = Agent(
    name="Researcher",
    instruction="We gather raw data and primary sources for the given topic.",
    model="gemini-2.0-flash"
)

# The Synthesizer (Step 2)
synthesizer = Agent(
    name="Synthesizer",
    instruction="We take the Researcher's output and create a formal report with executive summaries.",
    model="gemini-2.0-flash"
)

# --- 2. Create the Sequential Pipeline ---
# We use the SequentialAgent primitive for deterministic execution order
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
        # SequentialAgent executes its sub_agents in order
        response = await research_pipeline.run_async(input=request.topic)
        return {"status": "success", "report": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "alive"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
