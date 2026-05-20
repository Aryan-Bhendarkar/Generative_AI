from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="openai/gpt-4o-mini")

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summaries the following text {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

report_generation_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(str(x).split()) > 500, RunnableSequence(prompt2, model, parser)),
    (RunnablePassthrough())
)

final_chain = RunnableSequence(report_generation_chain, branch_chain)
result = final_chain.invoke({"topic": "Russia v/s Ukrian"})
print(result)