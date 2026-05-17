from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
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

prompt1 = tempalate1.invoke({'topic': 'blackhole'})
result1 = model.invoke(prompt1)

prompt2 = tempalate2.invoke({'text': result1.content})
result2 = model.invoke(prompt2)

print(result2.content)