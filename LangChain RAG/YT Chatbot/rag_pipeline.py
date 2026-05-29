"""Core RAG pipeline for the YouTube chatbot.

The four classic RAG stages (CampusX RAG lecture):
  1a. Ingestion     - fetch the transcript as plain text.
  1b. Splitting     - cut it into overlapping chunks.
  1c. Embed + Store - vectorise the chunks into a Chroma vector store.
  2.  Retrieval     - pull the most relevant chunks for a question.
  3.  Augmentation  - stuff those chunks into a prompt as `context`.
  4.  Generation    - the LLM answers using ONLY that context.

`build_chain()` wires stages 2-4 into one LangChain runnable so the UI can
just call `chain.invoke(question)`.
"""

import re

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

# Keep these consistent with the rest of the repo.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-4o-mini"

PROMPT = PromptTemplate.from_template(
    "You are a helpful assistant answering questions about a YouTube video.\n"
    "Answer ONLY from the transcript context provided below.\n"
    "If the context is insufficient to answer, say you don't know - "
    "do not make anything up.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def extract_video_id(url_or_id: str) -> str:
    """Return the 11-char video id from a YouTube URL, or pass an id through.

    Handles the common URL shapes: watch?v=, youtu.be/, /embed/, /shorts/.
    """
    patterns = [
        r"(?:v=|/embed/|/shorts/|youtu\.be/)([0-9A-Za-z_-]{11})",
        r"^([0-9A-Za-z_-]{11})$",  # already just an id
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract a video id from: {url_or_id!r}")


def fetch_transcript(url_or_id: str, languages=("en",)) -> str:
    """Fetch a video's transcript and return it as one plain-text string."""
    video_id = extract_video_id(url_or_id)

    ytt_api = YouTubeTranscriptApi()
    try:
        fetched = ytt_api.fetch(video_id, languages=list(languages))
    except TranscriptsDisabled:
        raise RuntimeError(f"Transcripts are disabled for video {video_id}.")
    except NoTranscriptFound:
        raise RuntimeError(
            f"No transcript found for {video_id} in languages {languages}."
        )
    except VideoUnavailable:
        raise RuntimeError(f"Video {video_id} is unavailable.")

    # Each snippet is a small timestamped chunk; join into one big string
    # because the next stage (text splitting) wants plain text, not timestamps.
    return " ".join(snippet.text for snippet in fetched)


def format_docs(retrieved_docs) -> str:
    """Stage 3 helper - merge the retrieved Document chunks into one string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_vector_store(transcript: str) -> Chroma:
    """Stages 1b + 1c - split the transcript and embed it into Chroma."""
    # chunk_overlap repeats a bit of text between chunks so a sentence cut at
    # a boundary still appears whole in at least one chunk.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def build_chain(url_or_id: str, k: int = 4):
    """Build the full RAG chain for one video.

    Returns a runnable: `chain.invoke("your question")` -> answer string.
    """
    transcript = fetch_transcript(url_or_id)
    vector_store = build_vector_store(transcript)

    # Stage 2 - the retriever turns a question into the top-k relevant chunks.
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    llm = ChatOpenRouter(model=LLM_MODEL, temperature=0.2)

    # Build the {context, question} dict the prompt expects:
    #   - context : retriever output -> format_docs (Stage 3)
    #   - question: the raw question, passed straight through
    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    # Stage 4 - the full chain: build inputs -> prompt -> LLM -> plain text.
    return parallel | PROMPT | llm | StrOutputParser()


if __name__ == "__main__":
    chain = build_chain("https://www.youtube.com/watch?v=Gfr50f6ZBvo")
    print(chain.invoke("What is the main topic of this video?"))
