from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model='openai/gpt-4o-mini')

chat_history: list[BaseMessage] = [
    SystemMessage(content="You are helpful AI Assistance")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    if(user_input == 'exit'):
        break

    result = model.invoke(chat_history)
    chat_history.append(result)
    print("AI", result.content)