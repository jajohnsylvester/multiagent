import os
import logging
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools import ToolContext
# New import for the built-in Web Server
from google.adk.server import Server 

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)

# 2. Define Shared Logic
def update_world_log(tool_context: ToolContext, observation: str) -> dict:
    logs = tool_context.state.get("world_log", [])
    logs.append(observation)
    tool_context.state["world_log"] = logs
    return {"status": "Observation recorded."}

# 3. Define the Agents
explorer = Agent(
    name="Explorer",
    model="gemini-2.5-flash",
    instruction="Find 2 unique facts about the topic and use 'update_world_log'.",
    tools=[update_world_log]
)

scribe = Agent(
    name="Scribe",
    model="gemini-2.5-flash",
    instruction="Read the 'world_log' in the state and write a brief legend."
)

# 4. Create the Multi-Agent Workflow
agent_world = SequentialAgent(
    name="MultiAgentWorld",
    sub_agents=[explorer, scribe]
)

# 5. Initialize the ADK Server
# This automatically enables the Web UI playground
app = Server(agent=agent_world)

if __name__ == "__main__":
    # Render provides the port; default to 10000 if not found
    port = int(os.environ.get("PORT", 10000))
    
    # Run the server on 0.0.0.0 so it's accessible externally
    app.run(host="0.0.0.0", port=port)
