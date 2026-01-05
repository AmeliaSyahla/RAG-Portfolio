import os
import io
import sys
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag. rag_core import (
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
                "pages": [],
                "display": doc_id
            })
    
    return formatted_refs


# ========== Page Configuration ==========
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ========== Custom CSS Styling ==========
def add_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* Main container - Dark theme */
        .stApp {
            background:  linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        . main . block-container {
            padding-top: 60px ! important;
            padding-bottom:  140px !important;
            max-width: 900px !important;
            margin: 0 auto;
        }
        
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        
        /* Top Navigation Bar */
        .top-navbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: rgba(26, 26, 46, 0.95);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            z-index: 1000;
        }
        
        .nav-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .nav-logo {
            font-size: 24px;
        }
        
        .nav-title {
            font-size: 18px;
            font-weight:  600;
            color: #ffffff;
            letter-spacing: -0.5px;
        }
        
        .nav-status {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius:  20px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .status-connected {
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .status-disconnected {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        . status-dot.connected {
            background: #10b981;
        }
        
        .status-dot.disconnected {
            background: #ef4444;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Welcome Screen */
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            text-align: center;
            padding: 40px 20px;
        }
        
        .welcome-icon {
            font-size: 64px;
            margin-bottom:  24px;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        .welcome-title {
            font-size: 32px;
            font-weight:  700;
            color: #ffffff;
            margin-bottom: 12px;
            letter-spacing: -1px;
        }
        
        .welcome-subtitle {
            font-size: 16px;
            color: rgba(255, 255, 255, 0.6);
            max-width: 400px;
            line-height: 1.6;
        }
        
        .welcome-tips {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: center;
            margin-top: 32px;
        }
        
        .tip-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px 20px;
            max-width: 200px;
            text-align:  left;
            transition: all 0.3s ease;
        }
        
        .tip-card:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        
        . tip-icon {
            font-size: 20px;
            margin-bottom: 8px;
        }
        
        .tip-text {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.7);
            line-height: 1.4;
        }
        
        /* Chat Container */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding: 20px 0;
        }
        
        /* Message Bubbles */
        .message-row {
            display: flex;
            gap: 12px;
            max-width: 85%;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform:  translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        . message-row.user {
            margin-left: auto;
            flex-direction: row-reverse;
        }
        
        .message-row.assistant {
            margin-right: auto;
        }
        
        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius:  50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            flex-shrink: 0;
        }
        
        .message-avatar. user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .message-avatar.assistant {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        
        .message-content {
            padding: 14px 18px;
            border-radius:  18px;
            line-height: 1.6;
            font-size: 14px;
        }
        
        .message-content.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            border-bottom-right-radius: 6px;
        }
        
        .message-content.assistant {
            background: rgba(255, 255, 255, 0.08);
            color: rgba(255, 255, 255, 0.9);
            border:  1px solid rgba(255, 255, 255, 0.1);
            border-bottom-left-radius: 6px;
        }
        
        . message-content.assistant strong {
            color: #10b981;
        }
        
        . message-content.assistant code {
            background: rgba(0, 0, 0, 0.3);
            padding: 2px 6px;
            border-radius:  4px;
            font-size: 13px;
        }
        
        /* Fixed Input Container */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(180deg, transparent 0%, rgba(26, 26, 46, 0.9) 20%, rgba(26, 26, 46, 1) 100%);
            padding: 20px 24px 24px;
            z-index: 999;
        }
        
        .input-wrapper {
            max-width: 850px;
            margin: 0 auto;
            position: relative;
        }
        
        .input-box {
            display: flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 4px;
            transition: all 0.3s ease;
        }
        
        .input-box:focus-within {
            border-color: rgba(102, 126, 234, 0.5);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Streamlit Input Overrides */
        .stTextInput > div > div > input {
            background: transparent ! important;
            border: none !important;
            color: #ffffff !important;
            font-size: 15px !important;
            padding:  14px 16px !important;
            border-radius: 12px !important;
        }
        
        .stTextInput > div > div > input:: placeholder {
            color: rgba(255, 255, 255, 0.4) !important;
        }
        
        .stTextInput > div > div > input:focus {
            box-shadow: none !important;
        }
        
        /* Button Styles */
        .stButton > button {
            border-radius: 12px ! important;
            padding: 10px 16px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            transition:  all 0.2s ease !important;
            border: none !important;
        }
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }
        
        .stButton > button[kind="primary"]:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        }
        
        .stButton > button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        /* Icon Buttons */
        .icon-btn {
            width: 44px;
            height: 44px;
            border-radius:  12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            cursor: pointer;
            transition: all 0.2s ease;
            border: none;
            background: transparent;
            color: rgba(255, 255, 255, 0.6);
        }
        
        .icon-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
        }
        
        .icon-btn. send {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .icon-btn. send:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .icon-btn.send:disabled {
            background: rgba(255, 255, 255, 0.1);
            color: rgba(255, 255, 255, 0.3);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.05);
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(102, 126, 234, 0.5);
            background: rgba(102, 126, 234, 0.05);
        }
        
        /* Dialog/Modal Styles */
        [data-testid="stModal"] > div {
            background: rgba(26, 26, 46, 0.98) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 20px !important;
            backdrop-filter: blur(20px) !important;
        }
        
        /* Selectbox */
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
            color: white !important;
        }
        
        /* Slider */
        .stSlider > div > div > div {
            background: rgba(255, 255, 255, 0.1) !important;
        }
        
        .stSlider > div > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        /* Checkbox */
        .stCheckbox label {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        /* Info/Warning/Error boxes */
        .stAlert {
            background: rgba(255, 255, 255, 0.05) !important;
            border:  1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px ! important;
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-color: #667eea transparent transparent transparent !important;
        }
        
        /* Scrollbar */
        : :-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        /* References styling */
        .references-box {
            margin-top: 12px;
            padding: 12px 16px;
            background: rgba(16, 185, 129, 0.1);
            border-left: 3px solid #10b981;
            border-radius: 0 8px 8px 0;
            font-size: 13px;
        }
        
        .references-title {
            color: #10b981;
            font-weight: 600;
            margin-bottom: 6px;
        }
        
        .reference-item {
            color: rgba(255, 255, 255, 0.7);
            padding: 2px 0;
        }
        
        /* Markdown in messages */
        .message-content.assistant h1,
        .message-content.assistant h2,
        .message-content.assistant h3 {
            color: #ffffff;
            margin-top: 16px;
            margin-bottom:  8px;
        }
        
        .message-content.assistant ul,
        .message-content. assistant ol {
            margin:  8px 0;
            padding-left: 20px;
        }
        
        . message-content.assistant li {
            margin: 4px 0;
        }
        
        .message-content.assistant p {
            margin: 8px 0;
        }
        
        /* Settings section headers */
        .settings-header {
            display: flex;
            align-items: center;
            gap: 8px;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 600;
            margin:  16px 0 12px;
        }
        
        . settings-header-icon {
            font-size: 18px;
        }
        
        /* Document list in settings */
        . doc-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            margin: 4px 0;
            font-size: 13px;
            color: rgba(255, 255, 255, 0.8);
        }
        
        .doc-icon {
            color: #667eea;
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
if "chunking_strategy" not in st. session_state:
    st. session_state.chunking_strategy = "recursive"
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


# ========== Load Chat History from Database on Startup ==========
if not st.session_state.chat_messages:
    history = load_chat_history(st.session_state.session_id)
    st.session_state.chat_messages = history
    if history:
        print(f"📚 Loaded {len(history)} messages from database")


# ========== Build Index Function ==========
def build_multimodal_index_from_files(
    files: List[io.BytesIO],
    embed_model: SentenceTransformer,
    vector_store: SupabaseVectorStore,
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
                        "page_number": page_num
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
                    "keywords": ", ".join(keywords[:3]),
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


# ========== Top Navigation Bar ==========
st.markdown(f"""
    <div class="top-navbar">
        <div class="nav-brand">
            <span class="nav-logo">📚</span>
            <span class="nav-title">RAG Assistant</span>
        </div>
        <div class="nav-status">
            <div class="status-pill {'status-connected' if st.session_state.supabase_connected else 'status-disconnected'}">
                <span class="status-dot {'connected' if st.session_state. supabase_connected else 'disconnected'}"></span>
                {'Connected • ' + str(st.session_state.total_chunks) + ' chunks' if st.session_state.supabase_connected else 'Disconnected'}
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Settings button row
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    if st.button("⚙️", key="settings_btn", help="Settings"):
        st.session_state.show_model_selector = True
with col3:
    if st.button("📤", key="upload_btn", help="Upload Documents"):
        st.session_state.show_settings = True


# ========== Model Selector Modal ==========
if st.session_state.show_model_selector:
    @st.dialog("⚙️ Settings")
    def show_model_settings():
        st.markdown("#### 🤖 Model Selection")
        chat_model = st.selectbox(
            "Chat Model",
            options=["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"],
            index=["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"]. index(st.session_state. chat_model) if st.session_state.chat_model in ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"] else 0,
        )
        st.session_state.chat_model = chat_model
        
        st.markdown("---")
        st.markdown("#### 🎯 Retrieval Settings")
        
        st.session_state.enable_reranking = st.checkbox(
            "Enable Reranking", 
            value=st.session_state.enable_reranking,
            help="Rerank results for better relevance"
        )
        
        if st.session_state.enable_reranking:
            st. session_state.rerank_method = st.selectbox(
                "Reranking Method",
                options=["hybrid", "keyword", "semantic"],
                index=["hybrid", "keyword", "semantic"].index(st.session_state.rerank_method)
            )
            st.session_state.rerank_top_k = st.slider(
                "Initial Retrieval Count", 
                10, 50, 
                st.session_state.rerank_top_k, 
                5,
                help="Number of documents to retrieve before reranking"
            )
        
        st.session_state.top_k = st.slider(
            "Final Results", 
            1, 10, 
            st.session_state.top_k,
            help="Number of documents to use for answer generation"
        )
        
        st.markdown("---")
        st.markdown("#### 🗄️ Knowledge Base")
        
        if st.session_state.vector_store: 
            doc_count = st.session_state.vector_store.get_document_count()
            st.info(f"📊 **{doc_count}** chunks indexed")
            
            doc_ids = st.session_state.vector_store.get_all_doc_ids()
            if doc_ids:
                st.markdown("**Indexed Documents:**")
                for doc_id in doc_ids[: 5]: 
                    st.markdown(f"""
                        <div class="doc-item">
                            <span class="doc-icon">📄</span>
                            <span>{doc_id}</span>
                        </div>
                    """, unsafe_allow_html=True)
                if len(doc_ids) > 5:
                    st.caption(f"... and {len(doc_ids) - 5} more")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✓ Save", use_container_width=True, type="primary"):
                st.session_state.show_model_selector = False
                st.rerun()
        with col2:
            if st. button("🗑️ Clear Chat", use_container_width=True):
                clear_chat_history(st.session_state.session_id)
                st.session_state.chat_messages = []
                st. session_state.show_model_selector = False
                st.rerun()
        with col3:
            if st. button("🧹 Clear Docs", use_container_width=True):
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
            if st.button("Close", use_container_width=True):
                st.session_state.show_settings = False
                st. rerun()
            return
        
        st.markdown("Upload PDF or DOCX files to build your knowledge base.")
        
        if st.session_state.vector_store:
            doc_ids = st.session_state.vector_store.get_all_doc_ids()
            if doc_ids:
                with st.expander(f"📚 Existing Documents ({len(doc_ids)})", expanded=False):
                    for doc_id in doc_ids:
                        st.markdown(f"• {doc_id}")
        
        uploaded_files = st.file_uploader(
            "Drop files here or click to browse",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.success(f"✓ {len(uploaded_files)} file(s) selected")
            
            col1, col2 = st. columns(2)
            with col1:
                if st.button("🔨 Process & Index", use_container_width=True, type="primary"):
                    with st.spinner("🔄 Processing documents...  This may take a while."):
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
                                st.balloons()
                                st.session_state.show_settings = False
                                st. rerun()
                            else: 
                                st.warning("⚠️ No new content added. Files may already exist.")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_settings = False
                    st.rerun()
        else:
            if st.button("Close", use_container_width=True):
                st.session_state.show_settings = False
                st.rerun()
    
    show_upload_dialog()


# ========== Chat Display ==========
if not st.session_state.chat_messages:
    # Welcome Screen
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">🤖</div>
            <div class="welcome-title">How can I help you today?</div>
            <div class="welcome-subtitle">
                Upload your documents and ask questions.  I'll find answers from your knowledge base.
            </div>
            <div class="welcome-tips">
                <div class="tip-card">
                    <div class="tip-icon">📄</div>
                    <div class="tip-text">Upload PDFs or DOCX files to get started</div>
                </div>
                <div class="tip-card">
                    <div class="tip-icon">💬</div>
                    <div class="tip-text">Ask questions in natural language</div>
                </div>
                <div class="tip-card">
                    <div class="tip-icon">🔗</div>
                    <div class="tip-text">I'll cite sources from your documents</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    # Chat Messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.chat_messages:
        if message["role"] == "user": 
            st.markdown(f"""
                <div class="message-row user">
                    <div class="message-avatar user">👤</div>
                    <div class="message-content user">{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Process assistant message for better display
            content = message["content"]
            # Convert markdown-style bold to HTML
            content = content.replace("**", "<strong>").replace("</strong><strong>", "**")
            
            st.markdown(f"""
                <div class="message-row assistant">
                    <div class="message-avatar assistant">🤖</div>
                    <div class="message-content assistant">{content}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ========== Fixed Input Area ==========
st.markdown('<div class="input-container"><div class="input-wrapper">', unsafe_allow_html=True)

col_input, col_send = st.columns([12, 1])

with col_input:
    query = st.text_input(
        "Message",
        placeholder="Ask anything about your documents...",
        label_visibility="collapsed",
        key="query_input"
    )

with col_send:
    ask_button = st.button("➤", key="send_btn", disabled=not query, type="primary")

st.markdown('</div></div>', unsafe_allow_html=True)


# ========== Query Processing ==========
if ask_button and query:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Not connected to Supabase. Please check your configuration.")
    elif st.session_state.total_chunks == 0:
        st.warning("⚠️ No documents indexed. Please upload documents first.")
    else:
        st.session_state.chat_messages.append({"role": "user", "content": query})
        
        with st.spinner("🔍 Searching and generating response..."):
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
                            ref_text = "\n\n📚 **References:**\n"
                            for ref in references: 
                                if ref['pages']:
                                    pages_str = ", ".join(map(str, ref['pages']))
                                    ref_text += f"• {ref['document']}, page {pages_str}\n"
                                else: 
                                    ref_text += f"• {ref['document']}\n"
                            answer_text += ref_text
                    
                    save_assistant_answer(session_id=session_id, answer=answer_text)
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer_text
                    })
                else:
                    answer_text = "❌ I couldn't find relevant information in the documents.  Try rephrasing your question or using different keywords."
                    save_assistant_answer(session_id=session_id, answer=answer_text)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer_text
                    })
                
                st.rerun()
                
            except Exception as e: 
                error_msg = f"❌ An error occurred: {str(e)}"
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content":  error_msg
                })
                import traceback
                print(traceback.format_exc())
                st.rerun()