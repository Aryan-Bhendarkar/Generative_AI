from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model='openai/gpt-4o-mini')

messages = [
    SystemMessage(content="You are a Helpful AI Assistance."), 
    HumanMessage(content="Tell me about Langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(result.content)
print(messages)