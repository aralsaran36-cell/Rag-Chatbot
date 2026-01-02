import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("📚 RAG Chat")

# Automatically clear old history on reload for testing
if "history" not in st.session_state:
    st.session_state.history = []

tabs = st.tabs(["💬 Chat", "📁 Upload Documents"])

# --- Chat tab ---
with tabs[0]:
    query = st.text_input("Enter your question:")

    if st.button("Send") and query.strip():
        # Append user query
        st.session_state.history.append(("user", query))
        try:
            # Call backend API
            resp = requests.post(f"{API_URL}/query", json={"query": query})
            answer = resp.json().get("answer", "(no response)")
            # Append only the LLM answer
            st.session_state.history.append(("assistant", answer))
        except Exception as e:
            st.error(f"Error: {e}")

    # Display chat
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

    # Optional: button to clear chat
    if st.button("Clear Chat"):
        st.session_state.history = []

# --- Upload tab ---
with tabs[1]:
    uploaded = st.file_uploader("Choose a file", type=["txt", "md"])
    if uploaded:
        try:
            files = {"file": (uploaded.name, uploaded.read())}
            resp = requests.post(f"{API_URL}/upload", files=files)
            st.success(f"Uploaded and indexed: {uploaded.name}")
        except Exception as e:
            st.error(f"Upload failed: {e}")

