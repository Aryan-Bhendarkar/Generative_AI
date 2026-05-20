from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="openai/gpt-4o-mini")

prompt1 = PromptTemplate(
    template="Generate the tweet about the {topic}", 
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate the Linkdin post about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkdin": RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': "AI"})
print(result)
print("Tweet: ", result["tweet"])
print("Linkdin: ", result["linkdin"])
