from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_query("Delhi is the capital of India.")
print(str(result))

# Embedding for Multiple documents 
documents = ["Delhi is the capital of India", "Kolkata is the Capital of West Bengal", "Paris is the capital of France"]
result2 = embedding.embed_documents(documents)
print(str(result2))