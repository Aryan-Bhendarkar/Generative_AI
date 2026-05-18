from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting things about {topic}",
    input_variables=['topic']
)

model = ChatOpenRouter(model='openai/gpt-4o-mini')

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})
print(result)

chain.get_graph().print_ascii()