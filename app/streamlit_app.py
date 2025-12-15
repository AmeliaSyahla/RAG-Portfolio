import os
import io
import sys
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.rag_core import (
    extract_elements_from_pdf,
    extract_elements_from_docx,
    summarize_text_with_groq,
    summarize_table_with_groq,
    summarize_image_with_gemini,
    extract_keywords_simple,
    generate_document_summary,
    build_vector_store_with_metadata,
    save_vector_store_with_metadata,
    load_vector_store_with_metadata,
    search_vector_store_with_reranking,
    answer_with_rag,
    DocChunk,
    DocumentMetadata,
    get_groq_client,
)

load_dotenv()

# ========== Helper Function ==========
def extract_references(chunks: List[DocChunk]) -> List[Dict[str, Any]]:
    """Extract unique references from retrieved chunks"""
    references = {}
    
    for chunk in chunks:
        doc_id = chunk.doc_id
        page_num = chunk. page_number if hasattr(chunk, 'page_number') else None
        
        if doc_id not in references: 
            references[doc_id] = set()
        
        if page_num:
            references[doc_id].add(page_num)
    
    # Format references
    formatted_refs = []
    for doc_id, pages in references.items():
        if pages:
            sorted_pages = sorted(list(pages))
            formatted_refs.append({
                "document":  doc_id,
                "pages": sorted_pages,
                "display": f"{doc_id}, halaman {', '.join(map(str, sorted_pages))}"
            })
        else:
            formatted_refs.append({
                "document": doc_id,
                "pages": [],
                "display": doc_id
            })
    
    return formatted_refs

# ========== Page Configuration ==========
st. set_page_config(
    page_title="RAG Chatbot - Multimodal",
    page_icon="📚",
    layout="wide"
)
# ========== Custom CSS Styling ==========
def add_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #0d1117;
            color: #e6edf3;
        }
        body {
            background-color: #0d1117;
        }
        [data-testid="stSidebar"] {
            background-color: #161b22;
        }
        . stButton>button {
            background-color: #238636;
            color: #ffffff;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #2ea043;
            color: white;
        }
        .stDataFrame {
            border-radius: 10px ! important;
            overflow: hidden !important;
        }
        .answer-box {
            background-color: #161b22;
            padding: 20px;
            border-radius:  12px;
            border: 1px solid #30363d;
            margin-top: 10px;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #58a6ff;
            font-weight: 700;
        }
        . score-badge {
            background-color: #238636;
            color: white;
            padding: 2px 8px;
            border-radius:  12px;
            font-size:  0.85em;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()

# ========== Title ==========
st.title("📚 RAG Chatbot - Multimodal")

# ========== Session State Initialization ==========
if "vector_store" not in st. session_state:
    st. session_state.vector_store = None
if "doc_chunks" not in st.session_state:
    st.session_state. doc_chunks = []
if "doc_metadata" not in st.session_state:
    st.session_state. doc_metadata = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ========== Load Embedding Model (Cached) ==========
@st.cache_resource
def load_embedding_model():
    """Load SentenceTransformer model (cached)"""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')

embed_model = load_embedding_model()
embed_model_name = 'sentence-transformers/all-mpnet-base-v2'

# ========== Sidebar Configuration ==========
with st.sidebar:
    st. header("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    
    col_build, col_reset = st.columns([2, 1])
    with col_build:
        build_button = st.button("🔨 Build Index", use_container_width=True)
    with col_reset:
        reset_button = st.button("🗑️", help="Clear index")
    
    st.markdown("---")
    st.header("⚙️ Chunking Settings")
    
    chunking_strategy = st.selectbox(
        "Strategy",
        options=["recursive", "semantic", "paragraph", "simple"],
        index=0,
        help="Choose chunking strategy"
    )
    
    chunk_size = st.slider("Chunk Size (chars)", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 50, 500, 100, 50)
    
    st.markdown("---")
    st.header("🎯 Retrieval Settings")
    
    enable_reranking = st.checkbox("Enable Reranking", value=True)
    
    if enable_reranking: 
        rerank_method = st.selectbox(
            "Reranking Method",
            options=["hybrid", "keyword", "semantic"],
            index=0,
            help="Hybrid:  70% semantic + 30% keyword"
        )
        rerank_top_k = st.slider("Initial Retrieval (before rerank)", 10, 50, 20, 5)
    else:
        rerank_method = "semantic"
        rerank_top_k = 5
    
    top_k = st.slider("Final Results", 1, 10, 5)
    
    st.markdown("---")
    st.header("🤖 Model Settings")
    
    chat_model_options = [
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    
    chat_model = st.selectbox(
        "GROQ Chat Model",
        options=chat_model_options,
        index=0,
    )
    
    st.markdown("---")
    st.info(f"🔧 **Embedding Model**\n\n{embed_model_name}")
    
    if st.session_state.vector_store is not None:
        st.success(f"✅ Index Loaded\n\n{st.session_state.total_chunks} chunks indexed")
    else:
        st.warning("⚠️ No index loaded")

# ========== Handle Reset Button ==========
if reset_button:
    st.session_state.vector_store = None
    st.session_state. doc_chunks = []
    st.session_state.doc_metadata = None
    st.session_state.total_chunks = 0
    st.rerun()

# ========== Build Index from Uploaded Files ==========
def build_multimodal_index_from_files(
    files: List[io.BytesIO],
    embed_model: SentenceTransformer,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int
):
    """Build vector store from uploaded files with multimodal support and page tracking"""
    documents = []
    doc_metadata_list = []
    groq_client = get_groq_client()
    
    for f in files:
        file_name = getattr(f, "name", "uploaded")
        file_ext = file_name.split('.')[-1].lower()
        
        # Save file temporarily
        temp_path = f"./temp_{file_name}"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(f.read())
        
        try: 
            # Extract elements based on file type (now with page tracking)
            if file_ext == "pdf":
                elements, images_base64, page_map = extract_elements_from_pdf(temp_path)
            elif file_ext == "docx": 
                elements, images_base64, page_map = extract_elements_from_docx(temp_path)
            else:
                st.warning(f"⚠️ Unsupported file type: {file_name}")
                continue
            
            if not elements:
                st. warning(f"⚠️ No content extracted from {file_name}")
                continue
            
            # Process text and tables
            texts = []
            tables = []
            
            for idx, element in enumerate(elements):
                element_type = str(type(element))
                if "Table" in element_type: 
                    tables.append((element, page_map.get(idx)))
                elif "CompositeElement" in element_type:
                    texts.append((element, page_map.get(idx)))
            
            # Create metadata
            all_text = " ".join([str(el) for el, _ in texts])
            keywords = extract_keywords_simple(all_text, top_n=5)
            summary = generate_document_summary(all_text, max_length=150)
            
            doc_metadata = DocumentMetadata(
                filename=file_name,
                file_size=len(all_text),
                creation_date=datetime. now(),
                page_count=len(elements),
                keywords=keywords,
                summary=summary,
                document_type=file_ext. upper(),
                doc_id=file_name
            )
            
            doc_metadata_list.append({
                "filename": file_name,
                "keywords": ", ".join(keywords),
                "summary": summary,
                "char_count": len(all_text),
                "page_count": len(elements),
                "content_types": f"Text:  {len(texts)}, Tables: {len(tables)}, Images: {len(images_base64)}"
            })
            
            chunk_id = 0
            
            # Process text chunks with page numbers
            for text_elem, page_num in texts: 
                text_content = str(text_elem)
                if text_content. strip():
                    text_summary = summarize_text_with_groq(text_content, groq_client)
                    
                    documents.append({
                        "text": text_summary,
                        "doc_id": file_name,
                        "chunk_id": chunk_id,
                        "category": "TEXT",
                        "keywords": ", ".join(keywords[: 3]),
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "content_type": "text",
                        "metadata": doc_metadata,
                        "page_number":  page_num
                    })
                    chunk_id += 1
            
            # Process tables with page numbers
            for table_elem, page_num in tables: 
                if hasattr(table_elem. metadata, 'text_as_html'):
                    table_html = table_elem.metadata. text_as_html
                    table_summary = summarize_table_with_groq(table_html, groq_client)
                    
                    documents.append({
                        "text": table_summary,
                        "doc_id": file_name,
                        "chunk_id": chunk_id,
                        "category": "TABLE",
                        "keywords": ", ".join(keywords[: 3]),
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "content_type": "table",
                        "metadata": doc_metadata,
                        "page_number": page_num
                    })
                    chunk_id += 1
            
            # Process images (images typically don't have page numbers in this extraction)
            for img_idx, img_base64 in enumerate(images_base64):
                img_description = summarize_image_with_gemini(img_base64)
                
                documents.append({
                    "text": img_description,
                    "doc_id": file_name,
                    "chunk_id": chunk_id,
                    "category": "IMAGE",
                    "keywords": ", ".join(keywords[:3]),
                    "chunk_size": chunk_size,
                    "chunk_overlap":  chunk_overlap,
                    "content_type": "image",
                    "metadata": doc_metadata,
                    "page_number": None
                })
                chunk_id += 1
        
        except Exception as e: 
            st.warning(f"⚠️ Failed to process {file_name}:  {str(e)}")
        finally:
            # Clean up temp file
            if os.path. exists(temp_path):
                os.remove(temp_path)
    
    if not documents:
        return None, None, None
    
    # Build vector store with page numbers
    vector_store, doc_chunks = build_vector_store_with_metadata(
        documents,
        embed_model,
        show_progress=True
    )
    
    return vector_store, doc_chunks, doc_metadata_list

# ========== Build Button Logic ==========
if build_button: 
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one file first.")
    else:
        with st.spinner("🔄 Building multimodal vector store..."):
            try:
                vector_store, doc_chunks, doc_metadata = build_multimodal_index_from_files(
                    uploaded_files,
                    embed_model,
                    chunking_strategy,
                    chunk_size,
                    chunk_overlap
                )
                
                if vector_store is not None:
                    st.session_state.vector_store = vector_store
                    st.session_state.doc_chunks = doc_chunks
                    st.session_state.doc_metadata = doc_metadata
                    st.session_state.total_chunks = len(doc_chunks)
                    
                    save_path = "./Memory"
                    save_vector_store_with_metadata(
                        vector_store, 
                        doc_chunks, 
                        save_path, 
                        "rag_multimodal_index"
                    )
                    
                    st.success(f"✅ Multimodal vector store built!  ({len(doc_chunks)} chunks)")
                    
                    with st.expander("📋 Document Summary", expanded=True):
                        summary_df = pd.DataFrame(doc_metadata)
                        st.dataframe(summary_df, use_container_width=True)
                else:
                    st.error("❌ No valid content found in uploaded files.")
            
            except Exception as e:
                st.error(f"❌ Error building index: {str(e)}")
                st.exception(e)

# ========== Auto-load Existing Vector Store ==========
if st.session_state.vector_store is None:
    try:
        if os.path.exists("./Memory/rag_multimodal_index"):
            with st.spinner("📂 Loading existing vector store..."):
                vector_store, metadata = load_vector_store_with_metadata(
                    "./Memory", 
                    "rag_multimodal_index", 
                    embed_model
                )
                st.session_state.vector_store = vector_store
                st. session_state.total_chunks = len(metadata)
                st.info(f"✅ Existing vector store loaded ({len(metadata)} chunks)")
    except Exception as e:
        pass

# ========== Chat Interface ==========
st.markdown("---")
st.header("💬 Ask Questions")

query = st.text_input(
    "Enter your question",
    placeholder="e.g., What is shown in the images?"
)

ask_button = st.button("🔍 Ask", use_container_width=False)

# ========== Query Processing ==========
answer_text = None
retrieved_results = []

if ask_button and query:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please build the index first / Silakan buat indeks terlebih dahulu dari sidebar.")
    else:
        with st.spinner("🔎 Searching multimodal content and generating answer..."):
            try:
                if enable_reranking:
                    results = search_vector_store_with_reranking(
                        st.session_state.vector_store,
                        query=query,
                        embed_model=embed_model,
                        k=top_k,
                        rerank_top_k=rerank_top_k,
                        rerank_method=rerank_method
                    )
                else:
                    search_results = st.session_state.vector_store.similarity_search_with_score(query, k=top_k)
                    results = []
                    for doc, score in search_results: 
                        results.append({
                            "text": doc. page_content,
                            "metadata": doc.metadata,
                            "score": float(score),
                            "combined_score":  float(score)
                        })
                
                if results:
                    retrieved_docs = [
                        DocChunk(
                            doc_id=r["metadata"]["doc_id"],
                            chunk_id=r["metadata"]["chunk_id"],
                            text=r["text"],
                            meta=r["metadata"],
                            content_type=r["metadata"]. get("content_type", "text"),
                            page_number=r["metadata"].get("page_number")
                        )
                        for r in results
                    ]
                    
                    # Generate answer with automatic language detection (no language parameter)
                    answer_text = answer_with_rag(query, retrieved_docs, chat_model)
                    retrieved_results = results
                    
                    # Extract and display references
                    references = extract_references(retrieved_docs)
                    st.session_state.references = references
                else:
                    answer_text = "❌ No relevant content found / Konten relevan tidak ditemukan."
            
            except Exception as e: 
                st.error(f"Error during search / Kesalahan saat pencarian: {str(e)}")
                st.exception(e)

# ========== Display Results ==========
if answer_text or retrieved_results:
    col1, col2 = st. columns([1, 1])
    
    with col1:
        st. subheader("💡 Answer / Jawaban")
        if answer_text:
            st.markdown(f"<div class='answer-box'>{answer_text}</div>", unsafe_allow_html=True)
            
            # Display references if available
            if hasattr(st.session_state, 'references') and st.session_state.references:
                st.markdown("---")
                st.markdown("**📚 Document References / Referensi Dokumen:**")
                for ref in st. session_state.references:
                    if ref['pages']:
                        pages_str = ", ".join(map(str, ref['pages']))
                        st.markdown(f"- **{ref['document']}**, halaman/page {pages_str}")
                    else: 
                        st.markdown(f"- **{ref['document']}**")
        else:
            st.caption("Answer will appear here / Jawaban akan muncul di sini.")
    
    with col2:
        st.subheader("📄 Retrieved Content / Konten yang Ditemukan")
        if retrieved_results:
            snippet_rows = []
            for rank, r in enumerate(retrieved_results, 1):
                content_type = r["metadata"].get("content_type", "text").upper()
                page_num = r["metadata"].get("page_number")
                page_display = f" (Hal.  {page_num})" if page_num else ""
                
                if enable_reranking and 'original_score' in r:
                    score_display = f"🎯 {r['combined_score']:.4f} (S:{r['original_score']:.3f} K:{r['keyword_score']:.3f})"
                else:
                    score_display = f"{r. get('combined_score', r['score']):.4f}"
                
                snippet_rows.append({
                    "Rank": rank,
                    "Type": content_type,
                    "Document": r["metadata"]["doc_id"] + page_display,
                    "Score": score_display,
                    "Content": r["text"][: 300] + ("..." if len(r["text"]) > 300 else "")
                })
            
            df_snippets = pd.DataFrame(snippet_rows)
            st.dataframe(df_snippets, use_container_width=True)
            
            csv_data = df_snippets.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Retrieved Content (CSV)",
                data=csv_data,
                file_name=f"retrieved_content_{datetime. now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("Retrieved content will appear here / Konten yang ditemukan akan muncul di sini.")

# ========== Reranking Info ==========
if enable_reranking and retrieved_results:
    with st.expander("ℹ️ Reranking Details"):
        st.markdown(f"""
        **Reranking Method:** `{rerank_method}`
        
        - **Initial Retrieval:** {rerank_top_k} candidates
        - **Final Results:** {top_k} documents
        - **Scoring:**
            - `S` = Semantic similarity (vector distance)
            - `K` = Keyword overlap score
            - `🎯` = Combined score (70% S + 30% K for hybrid)
        
        **Content Types:** Text, Tables, Images
        """)