from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model='openai/gpt-4o-mini')

class FeedBack(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiment of the provided feedback")

parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

classifier_clain = prompt1 | model | parser2
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),# type: ignore
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser), # type: ignore
    RunnableLambda(lambda x: "could not find sentiments") #Default Condition 
)

chain = classifier_clain | branch_chain

result = chain.invoke({'feedback': "This is terrible device"})
print(result)

chain.get_graph().print_ascii()