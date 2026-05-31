from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from dotenv import load_dotenv
import requests
import json
import os

load_dotenv()

API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")

@tool
def get_conversion_factor(base_currency:str, target_currency:str) -> float:
    """This function fetches the currency conversion factor between a given base currency and target currency"""

    url = f'https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()


rate = get_conversion_factor.invoke({"base_currency": 'USD', "target_currency": "INR"})
print(rate)


@tool
def convert(base_currency_value: int, conversion_rate:Annotated[float, InjectedToolArg]) -> float:
    """Given currency conversion rate, this function calculate the target currency value form the base currency value """
    return base_currency_value * conversion_rate



llm = ChatOpenRouter(model="openai/gpt-4o-mini")
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])


messages = [HumanMessage("What is the conversion factor between USD and INR, and based on that covert 10 USD to INR")]
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message) # type: ignore

for tool_call in ai_message.tool_calls:
    if tool_call["name"] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_message1.content)["conversion_rate"]
        messages.append(tool_message1)

    if tool_call["name"] == "convert":
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

result = llm_with_tools.invoke(messages)
print(result.content)