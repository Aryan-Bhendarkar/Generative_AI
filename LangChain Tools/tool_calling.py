from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def multiply(a:int, b:int) -> int:
    "Given Two numbers a and b, this tool will return their product"
    return a * b

# tool binding
llm = ChatOpenRouter(model="openai/gpt-4o-mini")
llm_with_tools = llm.bind_tools([multiply])

