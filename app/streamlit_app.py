import os
import io
import sys
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.rag_core import (
    extract_elements_from_pdf,
    extract_elements_from_docx,
    summarize_text_with_groq,
    summarize_table_with_groq,
    summarize_image_with_groq,
    extract_keywords_simple,
    generate_document_summary,
    search_vector_store_with_reranking,
    answer_with_rag,
    results_to_doc_chunks,
    DocChunk,
    DocumentMetadata,
    get_groq_client,
    SupabaseVectorStore,
)

from rag.history_chat import (
    process_user_query,
    save_assistant_answer,
    load_chat_history,
    clear_chat_history,
)

load_dotenv()


# ========== Helper Function ==========
def extract_references(chunks: List[DocChunk]) -> List[Dict[str, Any]]:
    """Extract unique references from retrieved chunks"""
    references = {}
    
    for chunk in chunks:
        doc_id = chunk.doc_id
        page_num = chunk.page_number if hasattr(chunk, 'page_number') else None
        
        if doc_id not in references:
            references[doc_id] = set()
        
        if page_num: 
            references[doc_id]. add(page_num)

    formatted_refs = []
    for doc_id, pages in references.items():
        if pages:
            sorted_pages = sorted(list(pages))
            formatted_refs.append({
                "document":  doc_id,
                "pages": sorted_pages,
                "display":  f"{doc_id}, halaman {', '.join(map(str, sorted_pages))}"
            })
        else:
            formatted_refs. append({
                "document": doc_id,
                "pages":  [],
                "display": doc_id
            })
    
    return formatted_refs


# ========== Page Configuration ==========
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ========== Custom CSS Styling ==========
def add_custom_css():
    st.markdown("""
        <style>
        /* ===== Import Fonts ===== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* ===== Global Styles ===== */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* ===== Hide Streamlit Elements ===== */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        
        /* ===== Main Container ===== */
        .stApp {
            background:  linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
        }
        
        . main . block-container {
            padding-top: 80px ! important;
            padding-bottom: 140px !important;
            padding-left: 5% !important;
            padding-right: 5% !important;
            max-width: 900px !important;
            margin: 0 auto !important;
        }
        
        /* ===== Top Navigation Bar ===== */
        .top-navbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 70px;
            background: rgba(26, 26, 46, 0.95);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            z-index:  1000;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        }
        
        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .navbar-logo {
            font-size: 28px;
        }
        
        .navbar-title {
            font-size: 20px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .navbar-status {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        /* ===== Status Badge ===== */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 50px;
            font-size:  13px;
            font-weight:  500;
            transition: all 0.3s ease;
        }
        
        .status-connected {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }
        
        .status-disconnected {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
            color: #fca5a5;
            border: 1px solid rgba(252, 165, 165, 0.3);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        . status-dot.connected {
            background: #34d399;
            box-shadow: 0 0 10px #34d399;
        }
        
        .status-dot.disconnected {
            background: #fca5a5;
            box-shadow: 0 0 10px #fca5a5;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* ===== Welcome Screen ===== */
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            text-align: center;
            padding: 40px 20px;
        }
        
        . welcome-icon {
            font-size: 80px;
            margin-bottom:  24px;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        .welcome-title {
            font-size: 36px;
            font-weight:  700;
            color: #ffffff;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .welcome-subtitle {
            font-size: 18px;
            color: rgba(255, 255, 255, 0.6);
            max-width: 500px;
            line-height: 1.6;
        }
        
        .welcome-hints {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: center;
            margin-top: 32px;
        }
        
        .hint-chip {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 10px 20px;
            border-radius:  50px;
            font-size:  14px;
            color: rgba(255, 255, 255, 0.7);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .hint-chip:hover {
            background:  rgba(102, 126, 234, 0.2);
            border-color: rgba(102, 126, 234, 0.5);
            color: #ffffff;
            transform: translateY(-2px);
        }
        
        /* ===== Chat Messages ===== */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            padding: 20px 0;
        }
        
        .message-wrapper {
            display: flex;
            flex-direction: column;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform:  translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message-wrapper. user {
            align-items: flex-end;
        }
        
        .message-wrapper.assistant {
            align-items: flex-start;
        }
        
        .message-label {
            font-size: 12px;
            font-weight: 500;
            margin-bottom: 6px;
            padding:  0 12px;
        }
        
        .message-label.user {
            color: #a78bfa;
        }
        
        .message-label.assistant {
            color: #60a5fa;
        }
        
        .message-bubble {
            max-width: 85%;
            padding: 16px 20px;
            border-radius:  20px;
            line-height: 1.7;
            font-size: 15px;
            box-shadow:  0 4px 20px rgba(0, 0, 0, 0.2);
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        
        . message-bubble.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            border-bottom-right-radius: 6px;
        }
        
        .message-bubble.assistant {
            background: rgba(255, 255, 255, 0.08);
            color: #e2e8f0;
            border:  1px solid rgba(255, 255, 255, 0.1);
            border-bottom-left-radius: 6px;
        }
        
        .message-bubble.assistant strong {
            color: #a78bfa;
        }
        
        .message-bubble.assistant code {
            background: rgba(0, 0, 0, 0.3);
            padding: 2px 6px;
            border-radius:  4px;
            font-size: 13px;
        }
        
        /* ===== Fixed Input Container (Background Blur) ===== */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 17.5%; /* Area aman di bawah */
            background: rgba(26, 26, 46, 0.95);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            z-index: 999;
        }

        /* ===== Input Bar Styling (The "Pill" Shape) ===== */
        /* Menargetkan baris Streamlit (HorizontalBlock) yang memiliki tombol upload */
        div[data-testid="stHorizontalBlock"]:has(.upload-btn-wrapper) {
            position: fixed;
            bottom: 25px; /* Jarak dari bawah */
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 850px;
            background: rgba(38, 39, 48); /* Warna background Pil */
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 8px 12px;
            z-index: 1000;
            align-items: center; /* Menyelaraskan item secara vertikal */
            gap: 8px !important; /* Jarak antar elemen */
        }

        /* ===== Membersihkan Style Bawaan Streamlit Input ===== */
        div[data-testid="stHorizontalBlock"]:has(.upload-btn-wrapper) .stTextInput > div > div {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.upload-btn-wrapper) .stTextInput > div > div > input {
            color: #ffffff !important;
            padding: 10px 0 !important; /* Mengurangi padding agar pas */
        }

        /* ===== Button Styling Adjustments ===== */
        /* Tombol Upload */
        .upload-btn-wrapper button {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            height: 42px !important;
            width: 42px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .upload-btn-wrapper button p {
            font-size: 20px !important;
            margin: 0 !important;
            padding-bottom: 2px !important; /* Koreksi vertikal icon */
        }

        /* Tombol Kirim */
        .send-btn-wrapper button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            height: 42px !important;
            width: 42px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .send-btn-wrapper button p {
            font-size: 20px !important;
            margin: 0 !important;
            color: white !important;
        }

        /* Sembunyikan gap bawaan kolom Streamlit agar rapat */
        div[data-testid="stHorizontalBlock"]:has(.upload-btn-wrapper) [data-testid="column"] {
            min-width: 0 !important;
            width: auto !important;
            flex: initial !important; /* Mencegah kolom mengambil lebar penuh */
        }

        /* Kolom tengah (Input) harus mengambil sisa ruang (flex-grow) */
        div[data-testid="stHorizontalBlock"]:has(.upload-btn-wrapper) [data-testid="column"]:nth-child(2) {
            flex: 1 !important;
            min-width: 200px !important;
        }
        
        /* Model Button */
        .model-btn button {
            background: rgba(255, 255, 255, 0.08) !important;
            border:  1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 50px !important;
            padding:  10px 20px !important;
            font-size: 13px !important;
            box-shadow: none !important;
        }
        
        .model-btn button:hover {
            background: rgba(255, 255, 255, 0.12) !important;
            border-color: rgba(255, 255, 255, 0.25) !important;
            transform:  none !important;
        }
        
        /* ===== Dialog/Modal Styles ===== */
        [data-testid="stModal"] > div {
            background: rgba(26, 26, 46, 0.98) !important;
            backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 20px !important;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5) !important;
        }
        
        [data-testid="stModal"] h1, 
        [data-testid="stModal"] h2, 
        [data-testid="stModal"] h3 {
            color: #ffffff !important;
        }
        
        [data-testid="stModal"] p,
        [data-testid="stModal"] label {
            color: rgba(255, 255, 255, 0.8) !important;
        }
        
        /* ===== File Uploader ===== */
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.05) !important;
            border:  2px dashed rgba(102, 126, 234, 0.4) !important;
            border-radius: 16px !important;
            padding: 30px !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(102, 126, 234, 0.7) !important;
            background: rgba(102, 126, 234, 0.1) !important;
        }
        
        [data-testid="stFileUploader"] label {
            color: rgba(255, 255, 255, 0.8) !important;
        }
        
        /* ===== Select Box ===== */
        . stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.08) !important;
            border:  1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
        }
        
        .stSelectbox > div > div:hover {
            border-color: rgba(102, 126, 234, 0.5) !important;
        }
        
        /* ===== Slider ===== */
        .stSlider > div > div > div {
            background: rgba(102, 126, 234, 0.3) !important;
        }
        
        .stSlider > div > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        /* ===== Checkbox ===== */
        .stCheckbox label {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        .stCheckbox > div[data-baseweb="checkbox"] > div {
            background: rgba(102, 126, 234, 0.8) !important;
            border-color: rgba(102, 126, 234, 0.8) !important;
        }
        
        /* ===== Info/Warning/Error Messages ===== */
        .stAlert {
            background: rgba(255, 255, 255, 0.05) !important;
            border:  1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px ! important;
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        [data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
        }
        
        /* ===== Scrollbar ===== */
        : :-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        
        : :-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.4);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(102, 126, 234, 0.6);
        }
        
        /* ===== Markdown Styles ===== */
        .stMarkdown {
            color: rgba(255, 255, 255, 0.9);
        }
        
        . stMarkdown h1, . stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #ffffff ! important;
        }
        
        .stMarkdown a {
            color: #667eea !important;
        }
        
        .stMarkdown code {
            background: rgba(0, 0, 0, 0.3) !important;
            color: #a78bfa !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
        }
        
        /* ===== Spinner ===== */
        .stSpinner > div {
            border-top-color: #667eea !important;
        }
        
        /* ===== Divider ===== */
        hr {
            border-color: rgba(255, 255, 255, 0.1) !important;
            margin: 20px 0 !important;
        }
        
        /* ===== Reference Box ===== */
        .reference-box {
            background: rgba(102, 126, 234, 0.1);
            border-left: 3px solid #667eea;
            padding:  12px 16px;
            border-radius: 0 8px 8px 0;
            margin-top: 12px;
            font-size: 13px;
            color: rgba(255, 255, 255, 0.7);
        }
        
        /* ===== Hide default column gaps ===== */
        [data-testid="column"] {
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()


# ========== Session State Initialization ==========
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "session_id" not in st.session_state:
    st.session_state.session_id = "default_user_session"
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "show_model_selector" not in st.session_state:
    st.session_state.show_model_selector = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_model" not in st.session_state:
    st.session_state.chat_model = "llama-3.1-8b-instant"
if "enable_reranking" not in st.session_state:
    st.session_state.enable_reranking = True
if "rerank_method" not in st.session_state:
    st.session_state.rerank_method = "hybrid"
if "rerank_top_k" not in st.session_state:
    st.session_state.rerank_top_k = 20
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "chunking_strategy" not in st.session_state:
    st.session_state.chunking_strategy = "recursive"
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 100
if "supabase_connected" not in st.session_state:
    st.session_state.supabase_connected = False


# ========== Load Embedding Model (Cached) ==========
@st.cache_resource
def load_embedding_model():
    """Load SentenceTransformer model (cached)"""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')


@st.cache_resource
def get_vector_store(_embed_model):
    """Get Supabase vector store (cached)"""
    try:
        store = SupabaseVectorStore(_embed_model)
        return store
    except Exception as e: 
        st.error(f"❌ Failed to connect to Supabase: {str(e)}")
        return None


embed_model = load_embedding_model()


# ========== Initialize Supabase Vector Store ==========
if st.session_state.vector_store is None:
    try:
        vector_store = get_vector_store(embed_model)
        if vector_store: 
            st.session_state. vector_store = vector_store
            st.session_state.total_chunks = vector_store.get_document_count()
            st.session_state.supabase_connected = True
            print(f"✅ Connected to Supabase with {st.session_state.total_chunks} chunks")
    except Exception as e:
        st.session_state.supabase_connected = False
        print(f"❌ Failed to connect to Supabase: {str(e)}")


# ========== Load Chat History from Database ==========
if not st.session_state.chat_messages:
    history = load_chat_history(st.session_state.session_id)
    st.session_state.chat_messages = history
    if history:
        print(f"📚 Loaded {len(history)} messages from database")


# ========== Build Index Function ==========
def build_multimodal_index_from_files(
    files: List[io.BytesIO],
    embed_model: SentenceTransformer,
    vector_store:  SupabaseVectorStore,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int
):
    """Build vector store from uploaded files"""
    documents = []
    doc_metadata_list = []
    groq_client = get_groq_client()
    
    for f in files:
        file_name = getattr(f, "name", "uploaded")
        file_ext = file_name.split('.')[-1].lower()
        
        if vector_store. document_exists(file_name):
            st.warning(f"⚠️ Document '{file_name}' already exists.  Skipping...")
            continue
        
        temp_path = f"./temp_{file_name}"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(f.read())
        
        try: 
            if file_ext == "pdf": 
                elements, images_base64, page_map = extract_elements_from_pdf(temp_path)
            elif file_ext == "docx":
                elements, images_base64, page_map = extract_elements_from_docx(temp_path)
            else:
                st.warning(f"⚠️ Unsupported file type: {file_name}")
                continue
            
            if not elements:
                st.warning(f"⚠️ No content extracted from {file_name}")
                continue
            
            texts = []
            tables = []
            
            for idx, element in enumerate(elements):
                element_type = str(type(element))
                if hasattr(element, 'element_type'):
                    if element.element_type == "table":
                        tables.append((element, page_map. get(idx)))
                    else:
                        texts.append((element, page_map.get(idx)))
                elif "Table" in element_type:
                    tables.append((element, page_map.get(idx)))
                elif "CompositeElement" in element_type: 
                    texts.append((element, page_map.get(idx)))
                else:
                    texts.append((element, page_map.get(idx)))
            
            all_text = " ".join([str(el) for el, _ in texts])
            keywords = extract_keywords_simple(all_text, top_n=5)
            
            doc_metadata_list.append({
                "filename": file_name,
                "keywords": ", ".join(keywords),
                "content_types": f"Text:  {len(texts)}, Tables: {len(tables)}, Images: {len(images_base64)}"
            })
            
            chunk_id = 0
            
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
                        "filename": file_name,
                        "page_number":  page_num
                    })
                    chunk_id += 1
            
            for table_elem, page_num in tables: 
                table_content = str(table_elem)
                if hasattr(table_elem, 'metadata') and hasattr(table_elem.metadata, 'text_as_html'):
                    table_content = table_elem.metadata.text_as_html
                
                if table_content.strip():
                    table_summary = summarize_table_with_groq(table_content, groq_client)
                    documents.append({
                        "text": table_summary,
                        "doc_id": file_name,
                        "chunk_id": chunk_id,
                        "category": "TABLE",
                        "keywords": ", ".join(keywords[:3]),
                        "chunk_size": chunk_size,
                        "chunk_overlap":  chunk_overlap,
                        "content_type": "table",
                        "filename": file_name,
                        "page_number": page_num
                    })
                    chunk_id += 1
            
            for img_idx, img_base64 in enumerate(images_base64):
                img_description = summarize_image_with_groq(img_base64, groq_client)
                documents.append({
                    "text": img_description,
                    "doc_id": file_name,
                    "chunk_id": chunk_id,
                    "category": "IMAGE",
                    "keywords": ", ".join(keywords[: 3]),
                    "content_type": "image",
                    "filename": file_name,
                    "page_number":  None
                })
                chunk_id += 1
        
        except Exception as e: 
            st.warning(f"⚠️ Failed to process {file_name}: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    if not documents:
        return 0, None
    
    total_added = vector_store.add_documents(documents, show_progress=True)
    return total_added, doc_metadata_list


# ========== Top Navigation Bar (HTML) ==========
status_class = "connected" if st.session_state. supabase_connected else "disconnected"
status_text = f"{st.session_state.total_chunks} chunks" if st.session_state.supabase_connected else "Offline"

st.markdown(f"""
    <div class="top-navbar">
        <div class="navbar-brand">
            <span class="navbar-title">RAG Chatbot</span>
        </div>
        <div class="navbar-status">
            <div class="status-badge status-{status_class}">
                <span class="status-dot {status_class}"></span>
                <span>{status_text}</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


# ========== Model Selector Button ==========
col_spacer1, col_model, col_spacer2 = st.columns([1, 2, 1])
with col_model:
    with st.container():
        st.markdown('<div class="model-btn">', unsafe_allow_html=True)
        if st.button(f"🤖 {st.session_state.chat_model}", key="model_btn", use_container_width=True):
            st.session_state.show_model_selector = not st.session_state. show_model_selector
        st.markdown('</div>', unsafe_allow_html=True)


# ========== Model Selector Modal ==========
if st.session_state.show_model_selector:
    @st.dialog("⚙️ Settings")
    def show_model_settings():
        st.markdown("#### 🤖 Model Selection")
        chat_model = st.selectbox(
            "Chat Model",
            options=["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0 if st.session_state.chat_model == "llama-3.1-8b-instant" else 1,
            label_visibility="collapsed"
        )
        st.session_state.chat_model = chat_model
        
        st.markdown("---")
        st.markdown("#### 🎯 Retrieval Settings")
        
        st.session_state.enable_reranking = st.checkbox(
            "Enable Reranking",
            value=st.session_state. enable_reranking
        )
        
        if st.session_state.enable_reranking:
            st. session_state.rerank_method = st.selectbox(
                "Reranking Method",
                options=["hybrid", "keyword", "semantic"],
                index=["hybrid", "keyword", "semantic"]. index(st.session_state. rerank_method)
            )
            st.session_state.rerank_top_k = st.slider(
                "Initial Retrieval",
                10, 50, st.session_state.rerank_top_k, 5
            )
        
        st.session_state. top_k = st.slider(
            "Final Results",
            1, 10, st.session_state.top_k
        )
        
        st. markdown("---")
        st.markdown("#### 🗄️ Vector Store")
        
        if st.session_state.vector_store: 
            doc_count = st.session_state.vector_store.get_document_count()
            st.info(f"📊 Total chunks: {doc_count}")
            
            doc_ids = st.session_state.vector_store.get_all_doc_ids()
            if doc_ids:
                st.markdown("**Indexed Documents:**")
                for doc_id in doc_ids[: 5]: 
                    st.markdown(f"• {doc_id}")
                if len(doc_ids) > 5:
                    st.caption(f"... and {len(doc_ids) - 5} more")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✓ Apply", use_container_width=True):
                st.session_state.show_model_selector = False
                st. rerun()
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_chat_history(st.session_state.session_id)
                st.session_state.chat_messages = []
                st.session_state. show_model_selector = False
                st.rerun()
        with col3:
            if st.button("🧹 Clear Docs", use_container_width=True):
                if st.session_state.vector_store:
                    with st.spinner("Clearing... "):
                        st.session_state.vector_store.delete_all()
                        st.session_state.total_chunks = 0
                    st.success("✅ Cleared!")
                    st.session_state.show_model_selector = False
                    st.rerun()
    
    show_model_settings()


# ========== Upload Modal ==========
if st.session_state.show_settings:
    @st.dialog("📤 Upload Documents")
    def show_upload_dialog():
        if not st.session_state.supabase_connected:
            st. error("❌ Not connected to Supabase.  Please check your configuration.")
            if st.button("Close"):
                st.session_state.show_settings = False
                st.rerun()
            return
        
        if st.session_state.vector_store: 
            doc_ids = st.session_state.vector_store.get_all_doc_ids()
            if doc_ids:
                st.info(f"📚 Existing documents: {', '.join(doc_ids[: 3])}{'...' if len(doc_ids) > 3 else ''}")
        
        uploaded_files = st.file_uploader(
            "Drop your PDF or DOCX files here",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files: 
            st.success(f"✓ {len(uploaded_files)} file(s) selected")
            
            col1, col2 = st. columns(2)
            with col1:
                if st.button("🔨 Build Index", use_container_width=True):
                    with st.spinner("🔄 Processing documents..."):
                        try:
                            total_added, doc_metadata = build_multimodal_index_from_files(
                                uploaded_files,
                                embed_model,
                                st.session_state.vector_store,
                                st.session_state. chunking_strategy,
                                st.session_state.chunk_size,
                                st.session_state.chunk_overlap
                            )
                            
                            if total_added > 0:
                                st.session_state.total_chunks = st.session_state.vector_store.get_document_count()
                                st.success(f"✅ Successfully added {total_added} chunks!")
                                st.session_state.show_settings = False
                                st.rerun()
                            else:
                                st. warning("⚠️ No new content added (files may already exist)")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_settings = False
                    st.rerun()
        else:
            if st.button("Close", use_container_width=True):
                st.session_state. show_settings = False
                st.rerun()
    
    show_upload_dialog()


# ========== Chat Display ==========
if not st.session_state.chat_messages:
    # Welcome Screen
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">🤖</div>
            <div class="welcome-title">Apa yang bisa saya bantu? </div>
            <div class="welcome-subtitle">
                Upload dokumen Anda dan mulai bertanya.  Saya akan membantu mencari informasi dari dokumen yang Anda berikan.
            </div>
            <div class="welcome-hints">
                <div class="hint-chip">📄 Upload PDF atau DOCX</div>
                <div class="hint-chip">💬 Tanya dalam Bahasa Indonesia</div>
                <div class="hint-chip">🔍 Cari informasi spesifik</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    # Chat Messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.chat_messages:
        if message["role"] == "user": 
            st.markdown(f"""
                <div class="message-wrapper user">
                    <div class="message-label user">You</div>
                    <div class="message-bubble user">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Process assistant message for better display
            content = message["content"]
            # Convert markdown bold to HTML
            content = content.replace("**", "<strong>").replace("</strong><strong>", "**")
            # Fix unclosed strong tags
            if content.count("<strong>") > content.count("</strong>"):
                content += "</strong>"
            
            st.markdown(f"""
                <div class="message-wrapper assistant">
                    <div class="message-label assistant">Assistant</div>
                    <div class="message-bubble assistant">{content}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Fixed Input Area ==========

# Layout Kolom Input
col_upload, col_input, col_send = st.columns([1, 14, 1])

with col_upload:
    # Class ini digunakan sebagai anchor untuk CSS selector
    st.markdown('<div class="upload-btn-wrapper">', unsafe_allow_html=True)
    if st.button("➕", key="plus_btn", help="Upload documents"):
        st.session_state.show_settings = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col_input:
    query = st.text_input(
        "Message",
        placeholder="Ketik pertanyaan Anda di sini...",
        label_visibility="collapsed",
        key="query_input"
    )

with col_send:
    st.markdown('<div class="send-btn-wrapper">', unsafe_allow_html=True)
    ask_button = st.button("↑", key="send_btn", disabled=not query, help="Send message")
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Query Processing ==========
if ask_button and query:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Not connected to Supabase.  Please check your configuration.")
    elif st.session_state.total_chunks == 0:
        st.warning("⚠️ No documents indexed.  Please upload documents first via + button.")
    else:
        st.session_state.chat_messages.append({"role":  "user", "content": query})
        
        with st.spinner("Thinking..."):
            try:
                session_id = st.session_state.session_id
                current_history = st.session_state.chat_messages[:-1]
                
                standalone_query = process_user_query(
                    session_id=session_id,
                    user_query=query
                )
                
                print(f"🔍 Searching with query: '{standalone_query}'")
                
                if st.session_state.enable_reranking:
                    results = search_vector_store_with_reranking(
                        st.session_state.vector_store,
                        query=standalone_query,
                        embed_model=embed_model,
                        k=st.session_state.top_k,
                        rerank_top_k=st.session_state.rerank_top_k,
                        rerank_method=st.session_state.rerank_method
                    )
                else: 
                    results = st.session_state.vector_store.similarity_search(
                        query=standalone_query,
                        k=st.session_state.top_k
                    )
                
                print(f"📊 Retrieved {len(results)} documents")
                
                if results:
                    retrieved_docs = results_to_doc_chunks(results)
                    
                    answer_text = answer_with_rag(
                        query=standalone_query,
                        retrieved=retrieved_docs,
                        chat_model=st.session_state.chat_model,
                        chat_history=current_history
                    )
                    
                    references = extract_references(retrieved_docs)
                    
                    if references:
                        has_refs = any(marker in answer_text for marker in ["Referensi:", "References:", "📚"])
                        if not has_refs:
                            ref_text = "\n\n📚 **Referensi:**\n"
                            for ref in references: 
                                if ref['pages']:
                                    pages_str = ", ".join(map(str, ref['pages']))
                                    ref_text += f"- {ref['document']}, halaman {pages_str}\n"
                                else:
                                    ref_text += f"- {ref['document']}\n"
                            answer_text += ref_text
                    
                    save_assistant_answer(session_id=session_id, answer=answer_text)
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer_text
                    })
                else:
                    answer_text = "❌ Tidak menemukan konten yang relevan dalam dokumen.  Coba ajukan pertanyaan dengan kata kunci yang lebih spesifik."
                    save_assistant_answer(session_id=session_id, answer=answer_text)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer_text
                    })
                
                st.rerun()
                
            except Exception as e: 
                error_msg = f"❌ Error: {str(e)}"
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content":  error_msg
                })
                import traceback
                print(traceback.format_exc())
                st.rerun()