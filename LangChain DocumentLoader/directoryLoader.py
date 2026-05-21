from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path=r"a:\AI-ML Track\GenAI\LangChain DocumentLoader\books",
    glob="*.pdf",
    loader_cls=PyPDFLoader # type: ignore
)

docs = loader.load()
print(docs[5])
print(docs[0].page_content)