import os
import io
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.rag_core import (
    embed_texts,
    chunk_text,
    build_faiss_index,
    answer_with_rag,
    DocChunk,
    get_groq_client,
)

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")

# Session state - INISIALISASI DI AWAL UNTUK MENCEGAH ERROR
if "index" not in st.session_state:
    st.session_state.index = None
if "meta" not in st.session_state:
    st.session_state.meta = None

# Load embedding model (cached)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')

embed_model = load_embedding_model()
embed_model_name = 'sentence-transformers/all-mpnet-base-v2' 

# Sidebar config
with st.sidebar:
    uploaded_pdfs = st.file_uploader(
        "Upload PDF resumes", type=["pdf"], accept_multiple_files=True
    )

    col_build, col_reset = st.columns([2, 1])
    with col_build:
        build_button = st.button("(Re)build Index")
    with col_reset:
        reset_button = st.button("🗑️ Reset", help="Clear index and start fresh")

    st.markdown("---")
    st.header("Settings")

    # Dropdown for Chat Models
    chat_model_options = [
    "openai/gpt-oss-20b",
    "llama3-70b-8192"
]
    chat_model = st.selectbox(
        "GROQ Chat Model",
        options=chat_model_options,
        index=chat_model_options.index("openai/gpt-oss-20b"),
    )

    top_k = st.slider("Top-K", 1, 10, 5)


# Session state sudah diinisialisasi di atas

# info model yang sedang digunakan untuk index
with st.sidebar:
    st.info(f"Using: {embed_model_name}")

# Handle reset button
if reset_button:
    st.session_state.index = None
    st.session_state.meta = None
    st.success("Index berhasil direset!")


def build_index_from_pdfs(files: List[io.BytesIO], embed_model: SentenceTransformer):
    rows = []
    for f in files:
        try:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            doc_id = getattr(f, "name", "uploaded.pdf")
            for i, ch in enumerate(chunk_text(text)):
                rows.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": i,
                        "text": ch,
                        "category": "Uploaded",
                    }
                )
        except Exception as e:
            st.warning(f"Failed to parse {getattr(f, 'name', 'PDF')}: {e}")
    if not rows:
        return None, None
    chunk_df = pd.DataFrame(rows)
    
    vecs = embed_texts(
        chunk_df["text"].tolist(),
        embed_model=embed_model,
        show_progress=True,
    )
    index = build_faiss_index(vecs)
    
    meta = {
        "doc_ids": chunk_df["doc_id"].tolist(),
        "chunk_ids": chunk_df["chunk_id"].tolist(),
        "texts": chunk_df["text"].tolist(),
        "categories": chunk_df["category"].tolist(),
    }
    return index, meta


if build_button:
    with st.spinner("Building index from uploaded PDFs..."):
        try:
            if uploaded_pdfs:
                index, meta = build_index_from_pdfs(uploaded_pdfs, embed_model)
                st.session_state.index = index
                st.session_state.meta = meta
                st.success(f"Index ready! Using {embed_model_name}")
            else:
                st.warning("Please upload at least one PDF to build the index.")
        except Exception as e:
            st.error(f"Error building index: {e}")

# Chat input
query = st.text_input(
    "Ask a question (e.g., *What is the content about?*)"
)
ask = st.button("Ask")

# Display
answer_text=None
retrieved_docs  = []
search_indices = None

if ask:
    if st.session_state.index is None or st.session_state.meta is None:
        answer_text = "⚠️ Build the index first from the sidebar."
    else:
        # Embed query
        qvec = embed_model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search FAISS index
        D, search_indices = st.session_state.index.search(qvec, int(top_k))
        
        # Retrieve documents
        meta = st.session_state.meta
        for idx in search_indices[0]:
            if idx < 0:
                continue
            retrieved_docs.append(
                DocChunk(
                    doc_id=meta["doc_ids"][idx],
                    chunk_id=meta["chunk_ids"][idx],
                    text=meta["texts"][idx],
                    meta={"category": meta["categories"][idx]},
                )
            )
        
        # Generate answer
        if retrieved_docs:
            answer_text = answer_with_rag(query, retrieved_docs, chat_model)
        else:
            answer_text = "No relevant documents found."

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Answer")
    if answer_text:
        st.write(answer_text)
    else:
        st.caption("Ask a question to see the answer here.")

with col2:
    st.subheader("Retrieved snippets")
    if search_indices is not None and st.session_state.meta:
        meta = st.session_state.meta
        rows = []
        for rank, idx in enumerate(search_indices[0]):
            if idx < 0:
                continue
            rows.append(
                {
                    "rank": rank + 1,
                    "doc_id": meta["doc_ids"][idx],
                    "category": meta["categories"][idx],
                    "snippet": meta["texts"][idx][:400]
                    + ("..." if len(meta["texts"][idx]) > 400 else ""),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.caption("No snippets retrieved.")
    else:
        st.caption(
            "Upload PDFs, build the index, and ask a question to see retrieved context here."
        )