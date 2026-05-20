from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="openai/gpt-4o-mini")

prompt = PromptTemplate(
    template="Write a joke on {topic}", 
    input_variables=['topic']
)

parser = StrOutputParser()

joke_generator_chain = RunnableSequence(prompt, model, parser)

# def word_count(word):
#     return len(word.split())

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    # "word_count": RunnableLambda(word_count)
    "word_count": RunnableLambda(lambda x: len(x.split()))
})

chain = RunnableSequence(joke_generator_chain, parallel_chain)
result = chain.invoke({"topic": "AI"})
print(result)