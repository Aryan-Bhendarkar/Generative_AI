from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents = [
    Document(page_content="Langchain helps developer to build LLM Applications easily"),
    Document(page_content="Chroma is a vectorDB optimized for LLM Based search"),
    Document(page_content="Embeddings convert text to high-dimensional vectors"),
    Document(page_content="OpenAI provides powerful embedding models")
]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents= documents,
    embedding= embedding_model,
    collection_name = "VectorStore-LLM"
)

retriver = vector_store.as_retriever(search_kwargs = {"k": 2})

query = "What is Chroma used for ?"
docs = retriver.invoke(query)

for i, doc in enumerate(docs):
    print(i)
    print(f"Content - \n {doc.page_content}")