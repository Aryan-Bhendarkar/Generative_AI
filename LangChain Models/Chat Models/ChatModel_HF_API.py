from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct"
) # pyright: ignore[reportCallIssue]

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is capital of India ?")
print(result.content)