import os

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model='openai/gpt-4o-mini')

result = model.invoke("What is capital of India?")

print(result)