"""Streamlit UI for the YouTube RAG chatbot.

Run it from this folder with:
    streamlit run app.py

Streamlit basics (since you're new to it):
  - The whole script re-runs top-to-bottom on EVERY interaction (button click,
    typing, etc.). That's the #1 thing to internalise.
  - So anything expensive (building the vector store) must be saved in
    `st.session_state` to survive re-runs instead of rebuilding every time.
  - `st.session_state` is just a dict that persists across re-runs per user.
"""

import streamlit as st

from rag_pipeline import build_chain

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎬")
st.title("🎬 YouTube RAG Chatbot")
st.caption("Paste a YouTube link, then ask questions about the video's content.")

# --- Session state: our "memory" across re-runs ---------------------------
# `chain` holds the built RAG pipeline; `messages` holds the chat history.
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: load a video ------------------------------------------------
with st.sidebar:
    st.header("1. Load a video")
    video_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    if st.button("Process video", type="primary"):
        if not video_url.strip():
            st.warning("Please paste a YouTube URL first.")
        else:
            # `st.spinner` shows a loading message while the block runs.
            with st.spinner("Fetching transcript and building the index..."):
                try:
                    st.session_state.chain = build_chain(video_url)
                    st.session_state.messages = []  # reset chat for new video
                    st.success("Video processed! Ask away. 👉")
                except Exception as err:
                    st.session_state.chain = None
                    st.error(f"Could not process video: {err}")

    if st.session_state.chain is not None:
        st.info("A video is loaded. Ask questions in the chat on the right.")

# --- Main area: the chat --------------------------------------------------
# Replay the conversation so far (history lives in session_state).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# `st.chat_input` renders a chat box pinned to the bottom and returns the
# typed text (or None if nothing was submitted this run).
question = st.chat_input("Ask something about the video...")

if question:
    if st.session_state.chain is None:
        st.warning("Load a video from the sidebar first.")
    else:
        # Show + store the user's message.
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate + show + store the assistant's answer.
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
