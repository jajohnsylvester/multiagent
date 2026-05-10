import os
import uvicorn
from fastapi import FastAPI
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import ToolContext
# CORRECTED IMPORT PATH BELOW
from google.adk.cli.fast_api import get_fast_api_app

# 1. Define your tool
def update_world_log(tool_context: ToolContext, observation: str) -> dict:
    logs = tool_context.state.get("world_log", [])
    logs.append(observation)
    tool_context.state["world_log"] = logs
    return {"status": "Logged to world state."}

# 2. Define the Agents
explorer = Agent(
    name="Explorer",
    model="gemini-2.5-flash",
    instruction="Find 2 facts about the topic and record them using 'update_world_log'.",
    tools=[update_world_log]
)

scribe = Agent(
    name="Scribe",
    model="gemini-2.5-flash",
    instruction="Read the 'world_log' from the state and write a short story."
)

# 3. Create the Orchestrator
agent_world = SequentialAgent(
    name="SequentialAgentWorld",
    sub_agents=[explorer, scribe]
)

# 4. Convert to FastAPI with Web UI enabled
# get_fast_api_app is located in google.adk.cli.fast_api
app = get_fast_api_app(agent_world, enable_ui=True)

# 5. Production execution for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # host MUST be 0.0.0.0 for Render to route traffic to your container
    uvicorn.run(app, host="0.0.0.0", port=port)
