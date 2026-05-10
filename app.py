import os
import uvicorn
from fastapi import FastAPI
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import ToolContext
# Use the high-level FastAPI utility
from google.adk.cli.fast_api import get_fast_api_app

# 1. Your Tool Logic
def update_world_log(tool_context: ToolContext, observation: str) -> dict:
    logs = tool_context.state.get("world_log", [])
    logs.append(observation)
    tool_context.state["world_log"] = logs
    return {"status": "Observation recorded."}

# 2. Agent Definitions
explorer = Agent(
    name="Explorer",
    model="gemini-2.5-flash", # Updated to current 2026 stable model
    instruction="Find 2 unique facts and use 'update_world_log'.",
    tools=[update_world_log]
)

scribe = Agent(
    name="Scribe",
    model="gemini-2.5-flash",
    instruction="Read 'world_log' from state and write a brief legend."
)

# 3. Multi-Agent Orchestrator
agent_world = SequentialAgent(
    name="SeqAgentWorld",
    sub_agents=[explorer, scribe]
)

# 4. Generate the FastAPI App
# Signature fix: Just pass the agent. The UI is bundled by default.
app = get_fast_api_app(agent_world)

# 5. Render Startup Logic
if __name__ == "__main__":
    # Render requires binding to 0.0.0.0 and the dynamic $PORT
    port = int(os.environ.get("PORT", 10000))
    print(f"AgentWorld active on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
