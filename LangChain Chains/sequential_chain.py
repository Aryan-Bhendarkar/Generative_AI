from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model = 'openai/gpt-4o-mini')

prompt1 = PromptTemplate(
    template="Generate the Detail report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 point summary from the following text \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': "Unemployment of India"})
print(result)

chain.get_graph().print_ascii()
