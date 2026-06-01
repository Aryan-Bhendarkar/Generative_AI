from langchain_openrouter import ChatOpenRouter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from dotenv import load_dotenv
import requests

load_dotenv()

search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke("today's top News in India ")

@tool
def get_weather(city: str) -> str:
    """This tool fetches the current weather data form a given city"""
    url = f"http://api.weatherstack.com/current?access_key=df739feb8122522cc2cdbae91af53a80&query={city}"
    response = requests.get(url)
    return response.json()

llm = ChatOpenRouter(model = "openai/gpt-4o-mini")


# Agent Building
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm = llm,
    tools = [search_tool, get_weather],
    prompt = prompt
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = [search_tool, get_weather],
    verbose = True
)

response = agent_executor.invoke({"input": "Find the second capital of Maharashtra and its current temperature."})
print(response)
