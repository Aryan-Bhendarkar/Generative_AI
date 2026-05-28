from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [
    Document(page_content="Langchain makes easy to work with LLM"),
    Document(page_content="Langchain is used to build LLM based applications"),
    Document(page_content="Chroma is used to store and search document embeddings"),
    Document(page_content="Embeddings are vector representation of text"),
    Document(page_content="MMR helps you to get diverse results when doing similarity search"),
    Document(page_content="Langchain supports Pinecone, Chroma, FAISS and more")
]

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {"k": 3, "lambda_mult": 0}
)

query = "What is LangChain ?"
results = retriever.invoke(query)

