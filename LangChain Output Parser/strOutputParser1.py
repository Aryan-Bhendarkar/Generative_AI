from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct"
)  # type: ignore

model = ChatHuggingFace(llm = llm)

tempalate1 = PromptTemplate(
    template='write a detail report on {topic}', 
    input_variables=['topic']
)

tempalate2 = PromptTemplate(
    template='write a 5 line summary on the following text. /n {text}', 
    input_variables=['text']
)

parser = StrOutputParser()

chain = tempalate1 | model | parser | tempalate2 | model | parser
result = chain.invoke({'topic': 'blackhole'})
print(result)