from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a:int, b:int) -> int:
    "Given Two numbers a and b, this tool will return their product"
    return a * b

# Tool binding
llm = ChatOpenRouter(model="openai/gpt-4o-mini")
llm_with_tools = llm.bind_tools([multiply])

# Tool calling 
query = HumanMessage("Multiply 3 with 6 and print the result")
messages = [query]
result = llm_with_tools.invoke(messages)
messages.append(result) # type: ignore
print(result)
print(result.tool_calls)

# Tool Execution
tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)

content = llm_with_tools.invoke(messages)
print("--" * 100)
print(content)