import os
import uvicorn
from fastapi import FastAPI
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import ToolContext
from google.adk.cli.fast_api import AdkWebServer

# 1. Define your tool
def update_world_log(tool_context: ToolContext, observation: str) -> dict:
    logs = tool_context.state.get("world_log", [])
    logs.append(observation)
    tool_context.state["world_log"] = logs
    return {"status": "Observation recorded."}

# 2. Define the Agents
explorer = Agent(
    name="Explorer",
    model="gemini-2.5-flash",
    instruction="Find 2 unique facts and use 'update_world_log'.",
    tools=[update_world_log]
)

scribe = Agent(
    name="Scribe",
    model="gemini-2.5-flash",
    instruction="Read 'world_log' from state and write a legend."
)

# 3. Create the Orchestrator
agent_world = SequentialAgent(
    name="SeqAgentWorld",
    sub_agents=[explorer, scribe]
)

# 4. Explicitly setup the ADK Web Server
# This is more stable than the internal 'get_fast_api_app' helper
server = AdkWebServer(agent=agent_world)
app = server.get_fast_api_app()

# 5. Add a simple health check for Render
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # Render requires binding to 0.0.0.0 and the $PORT variable
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
