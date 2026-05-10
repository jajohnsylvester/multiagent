import os
from fastapi import FastAPI
from google.adk.agents import Agent, SequentialAgent

app = FastAPI()

# 1. Define specialized sub-agents
researcher = Agent(
    name="Researcher",
    model="gemini-1.5-flash",
    instruction="Search for and provide 3-5 technical facts about the user's topic."
)

writer = Agent(
    name="Writer",
    model="gemini-1.5-flash",
    instruction="Using the facts provided by the Researcher, write a professional executive summary."
)

# 2. Create the Multi-Agent Coordinator (Sequential)
multi_agent_workflow = SequentialAgent(
    name="ResearchTeam",
    sub_agents=[researcher, writer]
)

# 3. Simple FastAPI endpoint to trigger the agents
@app.get("/chat")
async def chat(prompt: str):
    # The run() method executes the chain of agents
    response = multi_agent_workflow.run(prompt)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    # Render provides the port via an environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
