import os

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model='openai/gpt-4o-mini', temperature=1.5, max_completion_tokens=30)
result = model.invoke("What is capital of India? - Answer this questions creatively")

print(result.content)