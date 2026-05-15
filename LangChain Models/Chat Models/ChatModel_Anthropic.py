from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)

result = model.invoke("What is the Capital of India?")

print(result.content)

