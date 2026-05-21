from langchain_community.document_loaders import TextLoader
from langchain_openrouter import ChatOpenRouter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model='openai/gpt-4o-mini')

prompt = PromptTemplate(
    template="Write the summary for the following poem - \n {poem}",
    input_variables=["poem"]
)

parser = StrOutputParser()

loader = TextLoader("LangChain DocumentLoader/cricket.txt", encoding='utf-8')

docs = loader.load()
# # List 
# print(type(docs))
# print(docs)
# print(docs[0])
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser
result = chain.invoke({"poem": docs[0].page_content})
print(result)