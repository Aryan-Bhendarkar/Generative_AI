from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="openai/gpt-4o-mini")

prompt = PromptTemplate(
    template="Write Joke on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain this Joke - {joke}", 
    input_variables=["joke"]
)

parser = StrOutputParser()

joke_generator_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explaination": RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(joke_generator_chain, parallel_chain)
result = chain.invoke({"topic": "cricket"})
print(result)