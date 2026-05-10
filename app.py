import os
from typing import List
from fastapi import FastAPI
from google.adk.agents import Agent, ParallelAgent, ToolContext
from google.adk.llms import Gemini
import uvicorn

app = FastAPI()

# --- 1. Tools for interacting with the "World" ---

def update_world_log(tool_context: ToolContext, observation: str) -> dict:
    """Adds a new observation to the shared world log."""
    # Get existing logs or start empty
    logs = tool_context.state.get("world_log", [])
    logs.append(observation)
    # Save back to shared state
    tool_context.state["world_log"] = logs
    return {"status": "Observation recorded in the world."}

# --- 2. Define the Agents ---

# The Explorer: Gathers raw data/observations
explorer = Agent(
    name="Explorer",
    model="gemini-2.5-flash",
    instruction="""
    You are an explorer in a new digital world. 
    Your goal is to find 2 unique facts about the topic provided.
    Use the 'update_world_log' tool to record your findings.
    """,
    tools=[update_world_log]
)

# The Scribe: Synthesizes observations into a story
scribe = Agent(
    name="Scribe",
    model="gemini-2.5-flash",
    instruction="""
    You are the chronicler of the world.
    Wait for the Explorer to finish, then look at the 'world_log' in the state.
    Write a short, dramatic legend based on those findings.
    """
)

# --- 3. Orchestration (The World Engine) ---

# We use ParallelAgent so they both inhabit the same session state simultaneously
agent_world = ParallelAgent(
    name="AgentWorld",
    sub_agents=[explorer, scribe]
)

# --- 4. Web Endpoints ---

@app.get("/")
async def home():
    return {"status": "World is online", "endpoint": "/simulate?topic=Mars"}

@app.get("/simulate")
async def simulate(topic: str):
    # This triggers the multi-agent interaction
    result = agent_world.run(f"Begin exploration of: {topic}")
    return {
        "world_events": result,
        "final_state": agent_world.state  # Returns the shared 'world_log'
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
