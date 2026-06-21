import streamlit as st
import fitz  # pymupdf
import pymupdf4llm # NEW: Replaces pdfplumber
import re
import uuid
import json
import time
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# Custom premium styling (Glassmorphism, custom cards, and badges)
CSS_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

.rag-header {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.2rem;
    margin-bottom: 0.2rem;
    text-align: center;
}

.rag-sub-header {
    font-size: 1rem;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 2rem;
}

/* Custom Chunk Cards */
.chunk-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(8px);
}

.chunk-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 10px 20px rgba(99, 102, 241, 0.1);
}

.chunk-header {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
}

.chunk-badge {
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
}

/* Status Badges */
.chunk-badge.pending {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.25);
}

.chunk-badge.ingested {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
}

.chunk-badge.error {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.25);
}

.chunk-badge.page {
    background: rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.chunk-badge.type {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.25);
}

.chunk-badge.score {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #ffffff;
}

.chunk-badge.section {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

.chunk-body {
    font-size: 0.925rem;
    color: #cbd5e1;
    line-height: 1.6;
}

/* Tag style */
.tag-container {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}

.keyword-tag {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    color: #94a3b8;
}
</style>
"""

def extract_json_array(llm_output: str, show_error: bool = True) -> list[dict]:
    """Extract and parse a JSON array from the raw LLM output, with fallbacks for common syntax issues."""
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', llm_output, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = llm_output.strip()
        
    start_idx = json_str.find('[')
    end_idx = json_str.rfind(']')
    if start_idx != -1 and end_idx != -1:
        json_str = json_str[start_idx:end_idx+1]
        
    try:
        # Fallback 1: Direct JSON parsing
        data = json.loads(json_str, strict=False)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except Exception as e1:
        # Fallback 2: Try stripping trailing commas which often fail standard JSON parsers
        try:
            cleaned_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            data = json.loads(cleaned_str, strict=False)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except Exception:
            pass

        # Fallback 3: Try ast.literal_eval for Python literal syntax (e.g., single-quoted strings)
        try:
            import ast
            data = ast.literal_eval(json_str)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except Exception:
            pass

        if show_error:
            st.error(f"Failed to parse JSON array from LLM response: {e1}")
    return []


def render_single_chunk_card(chunk, idx, is_live=False):
    meta = chunk.get("metadata", {})
    title = chunk.get("title", "Untitled")
    text = chunk.get("text", "")
    item_type = meta.get("item_type", "N/A")
    item_id = meta.get("item_id") or "N/A"
    page = meta.get("page", 1)
    keywords = meta.get("keywords", [])
    status = chunk.get("status", "pending")
    
    status_label = "Pending"
    if status == "ingested":
        status_label = "Ingested"
    elif status == "error":
        status_label = "Error"
        
    safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
    safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
    safe_item_id = item_id.replace("<", "&lt;").replace(">", "&gt;")
    
    tags_html = "".join([f'<span class="keyword-tag">{kw}</span>' for kw in keywords])
    
    card_html = f"""
    <div class="chunk-card">
        <div class="chunk-header">
            <span class="chunk-badge {status}">{status_label}</span>
            <span class="chunk-badge page">Page {page}</span>
            <span class="chunk-badge type">{item_type}</span>
            <span class="chunk-badge page">ID: {safe_item_id}</span>
        </div>
        <div class="chunk-body">
            <strong style="font-size:1.05rem; color:#f1f5f9;">{safe_title}</strong>
            <p style="margin-top:8px; margin-bottom:8px;">{safe_text}</p>
        </div>
        <div class="tag-container">
            {tags_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if status == "pending":
            if is_live:
                st.markdown("<span style='color:#fbbf24; font-weight:600; padding:6px 0; display:inline-block;'>⏳ Ingestion pending</span>", unsafe_allow_html=True)
            else:
                if st.button(f"📥 Ingest Chunk {idx+1}", key=f"ingest_indiv_{chunk['id']}", use_container_width=True):
                    try:
                        st.session_state.rag.ingest_chunk(
                            collection_name=st.session_state.target_collection_name,
                            chunk_id=chunk["id"],
                            text=text,
                            title=title,
                            metadata=meta
                        )
                        chunk["status"] = "ingested"
                        st.success(f"Ingested Chunk {idx+1}!")
                        st.rerun()
                    except Exception as e:
                        chunk["status"] = "error"
                        st.error(f"Failed to ingest: {e}")
        elif status == "ingested":
            st.markdown("<span style='color:#34d399; font-weight:600; padding:6px 0; display:inline-block;'>✓ Added to collection</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#f87171; font-weight:600; padding:6px 0; display:inline-block;'>✗ Ingestion error</span>", unsafe_allow_html=True)

# Helper functions convert_table_to_markdown and extract_page_content are deleted since pymupdf4llm handles both natively.

def safe_get_response(active_llm, messages, stream=False, max_retries=5):
    """Helper to call LLMManager.get_response with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return active_llm.get_response(messages, stream=stream)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "Too Many Requests" in err_msg or "rate limit" in err_msg.lower():
                wait_time = 2 ** attempt
                print(f"[UI/rag_tab] Rate limit hit. Retrying get_response in {wait_time}s... (Attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded for chat response.")
def merge_small_chunks(chunks: list[dict], threshold_chars: int = 1600) -> list[dict]:
    if not chunks:
        return []
    
    merged_chunks = []
    current_chunk = None
    
    for c in chunks:
        if not isinstance(c, dict):
            continue
        
        c_text = c.get("text", "")
        
        if current_chunk is None:
            current_chunk = c
            continue
            
        current_text = current_chunk.get("text", "")
        
        # Check if merging would exceed the threshold
        if len(current_text) + len(c_text) + 2 <= threshold_chars:
            # Merge text
            current_chunk["text"] = current_text + "\n\n" + c_text
            
            # Merge title
            t1 = current_chunk.get("title", "Untitled")
            t2 = c.get("title", "Untitled")
            if t1 == t2:
                current_chunk["title"] = t1
            else:
                combined_title = f"{t1} / {t2}"
                if len(combined_title) > 100:
                    current_chunk["title"] = t1
                else:
                    current_chunk["title"] = combined_title
            
            # Merge metadata safely
            meta1 = current_chunk.get("metadata", {})
            meta2 = c.get("metadata", {})
            merged_meta = meta1.copy()
            
            # keywords union
            k1 = meta1.get("keywords", [])
            k2 = meta2.get("keywords", [])
            if isinstance(k1, list) and isinstance(k2, list):
                merged_meta["keywords"] = list(set(k1 + k2))
            
            # item_id
            id1 = meta1.get("item_id")
            id2 = meta2.get("item_id")
            if id1 and id2 and id1 != id2:
                merged_meta["item_id"] = f"{id1}, {id2}"
            elif id2:
                merged_meta["item_id"] = id2
                
            # item_name
            name1 = meta1.get("item_name")
            name2 = meta2.get("item_name")
            if name1 and name2 and name1 != name2:
                merged_meta["item_name"] = f"{name1} & {name2}"
            elif name2:
                merged_meta["item_name"] = name2
                
            current_chunk["metadata"] = merged_meta
        else:
            # Merging exceeds threshold, save current_chunk and start new one
            merged_chunks.append(current_chunk)
            current_chunk = c
            
    if current_chunk is not None:
        merged_chunks.append(current_chunk)
        
    return merged_chunks

def generate_chunks_with_llm(page_markdown: str, page_num: int, llm=None) -> list[dict]:
    """Chunk layout-aware Markdown into technical specification entries."""
    if not page_markdown.strip():
        return []

    system_prompt = (
        "You are an expert systems engineer and technical writer. Your task is to read a Markdown-formatted page "
        "extracted from a technical manual/specification and break it down into high-quality, standalone text chunks "
        "optimized for semantic search and RAG.\n\n"
        "Instructions:\n"
        "1. STRICT SIZE LIMIT: Each chunk's text MUST be strictly limited to a maximum of 250 to 300 words (or ~1000-1200 characters) to ensure it fits well within the embedding model's 512-token limit.\n"
        "2. LOGICAL CHUNKING & SPLITTING: Group related paragraphs, requirements, lists, and tables. If a page or topic is short, represent it in a single chunk. If a page has too much content and exceeds 250-300 words, you MUST split it into multiple chunks.\n"
        "3. CONNECTING CONTEXT: When a topic or page is split into multiple chunks, you MUST include clear connecting context in the subsequent chunks. Prepend or weave in information pointing back to the main topic or parent section (e.g., prefixing with '[Continued from {topic} / Section {name}]') so that the subsequent chunk does not lose semantic meaning when retrieved independently.\n"
        "4. TABLE RULE: Do not aggressively split tables row-by-row into tiny fragments. Keep tables intact if they fit within the 250-300 word limit. If a table is extremely large and must be split, group logical blocks of rows and include the table header and topic context in each split part.\n"
        "5. Preserve the structure of Markdown tables, lists, or structured data within the text, and assign appropriate metadata to each chunk.\n and remove any copyright or disclaimer or legal  related stuff donot add them to chunks or create new ones for them."
        "6. Output the result ONLY as a valid JSON array of objects. Do not include any commentary outside the JSON.\n\n"
        "Each object in the JSON array must follow this exact schema:\n"
        "[\n"
        "  {\n"
        "    \"title\": \"A descriptive title of the topic (e.g. 'Topic Name (Part 2)')\",\n"
        "    \"text\": \"The detailed content under 250 words. If this is a continuation, start with clear connecting context referencing the parent topic.\",\n"
        "    \"metadata\": {\n"
        "      \"item_type\": \"requirement\", \"configuration\", \"architecture\", \"api\", \"definition\", or \"other\",\n"
        "      \"item_id\": \"Any specific ID found (e.g., SWS_Can_00123, R12, C4) or null\",\n"
        "      \"item_name\": \"The title or name of the requirement/topic\",\n"
        "      \"keywords\": [\"kw1\", \"kw2\", \"kw3\"]\n"
        "    }\n"
        "  }\n"
        "]"
    )

    user_prompt = f"Page Number: {page_num}\n\nMarkdown Content:\n{page_markdown}"

    active_llm = llm if llm is not None else st.session_state.llm

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = safe_get_response(active_llm, messages, stream=False)
        llm_output = response.choices[0].message.content
        chunks = extract_json_array(llm_output, show_error=False)
        
        if not chunks:
            # Optimize prompt sizes to strictly fit within the 512 input token limit.
            # Avoid sending the original markdown user_prompt again; the model only needs to fix the JSON syntax of its output.
            retry_system_prompt = (
                "You are an expert JSON repair assistant. Fix the JSON syntax of the input and return ONLY a valid JSON array "
                "conforming to the schema: each object has 'title', 'text', and 'metadata' (with 'item_type', 'item_id', 'item_name', 'keywords'). "
                "Do not include any introductory or concluding text outside the JSON block."
            )
            # Truncate llm_output if it's exceptionally long to protect the token budget
            max_failed_chars = 1200
            truncated_output = llm_output if len(llm_output) <= max_failed_chars else llm_output[:max_failed_chars] + "... [truncated]"
            correction_user_prompt = (
                f"--- Incorrect JSON Attempt ---\n{truncated_output}\n\n"
                "CRITICAL: The JSON syntax above is invalid. Please rewrite it as a clean, valid JSON array conforming to the schema. "
                "Ensure all quotes, brackets, and commas are correct."
            )
            messages_retry = [
                {"role": "system", "content": retry_system_prompt},
                {"role": "user", "content": correction_user_prompt}
            ]
            print(f"[UI/rag_tab] JSON parsing failed. Retrying with self-correction prompt...", flush=True)
            try:
                response_retry = safe_get_response(active_llm, messages_retry, stream=False)
                llm_output_retry = response_retry.choices[0].message.content
                chunks = extract_json_array(llm_output_retry, show_error=True)
            except Exception as retry_err:
                print(f"[UI/rag_tab] Self-correction API call failed: {retry_err}", flush=True)

        if chunks:
            for c in chunks:
                if not isinstance(c, dict):
                    continue
                if "metadata" not in c or not isinstance(c["metadata"], dict):
                    c["metadata"] = {}
                c["metadata"]["page"] = page_num
                
                if "text" not in c:
                    for key in ["description", "content", "body"]:
                        if key in c:
                            c["text"] = c[key]
                            break
                if "text" not in c:
                    c["text"] = json.dumps(c)
            chunks = merge_small_chunks(chunks, threshold_chars=1600)
            return chunks
    except Exception as e:
        print(f"[UI/rag_tab] LLM chunk generation error: {e}", flush=True)
            
    return [{
        "title": f"Page {page_num} Raw Content",
        "text": page_markdown,
        "metadata": {
            "item_type": "page_text",
            "item_id": None,
            "item_name": f"Raw content of page {page_num}",
            "keywords": ["fallback"],
            "page": page_num
        }
    }]

# Dialog decorator for target collection settings and extraction parameters
@st.dialog("Configure Target Collection & Parameters")
def configure_target_collection_dialog(file_obj):
    st.write(f"📂 **File Uploaded:** `{file_obj.name}`")
    
    # 1. Target Collection Section
    st.markdown("### 🗃️ 1. Target Collection")
    collection_mode = st.radio(
        "Choose target option:",
        ["Add to Existing Collection", "Create New Collection"],
        key="dlg_collection_mode"
    )
    
    target_collection = ""
    existing_cols = st.session_state.rag.get_collections()
    
    if collection_mode == "Add to Existing Collection":
        if existing_cols:
            target_collection = st.selectbox("Select Target Collection:", existing_cols, key="dlg_select_col")
        else:
            st.warning("No existing collections found. Please select 'Create New Collection'.")
    else:
        raw_name = st.text_input("Enter New Collection Name:", placeholder="e.g. autosar_manuals", key="dlg_new_col_name").strip()
        target_collection = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name) if raw_name else ""
        if raw_name and target_collection != raw_name:
            st.info(f"Cleaned collection name to: `{target_collection}`")
            
    # 2. Extraction Parameters Section
    is_pdf = file_obj.name.lower().endswith(".pdf")
    total_pages = 0
    start_page = 1
    end_page = 1
    extract_ready = True
    
    if is_pdf:
        st.markdown("### ⚙️ 2. PDF Extraction Parameters")
        try:
            pdf_bytes = file_obj.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            file_obj.seek(0)
        except Exception as e:
            total_pages = 0
            st.error(f"Error reading PDF: {e}")
            extract_ready = False
            
        if total_pages > 0:
            st.info(f"Loaded specification contains {total_pages} pages.")
            col_start, col_end = st.columns(2)
            with col_start:
                start_page = st.number_input("Start Page:", min_value=1, max_value=total_pages, value=1, key="dlg_start_page")
            with col_end:
                end_page = st.number_input("End Page:", min_value=1, max_value=total_pages, value=total_pages, key="dlg_end_page")
                
            if start_page > end_page:
                st.error("Start Page cannot be greater than End Page.")
                extract_ready = False
        else:
            extract_ready = False
    else:
        st.markdown("### ⚙️ 2. File Parameters")
        if file_obj.name.lower().endswith(('.xlsx', '.xls')):
            st.info("Excel binary row parsing will be used.")
        else:
            st.info("Direct text content parsing will be used.")

    if st.button("🧱 Confirm & Load Chunks", type="primary", disabled=not extract_ready, key="dlg_confirm_btn"):
        if not target_collection:
            st.error("Please enter/select a valid collection name.")
        else:
            st.session_state.target_collection_name = target_collection
            st.session_state.collection_mode = collection_mode
            st.session_state.start_page = int(start_page)
            st.session_state.end_page = int(end_page)
            st.session_state.dialog_completed = True
            st.session_state.trigger_extraction_phase = True
            st.rerun()

def render_rag_tab():
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    
    st.markdown('<div class="rag-header">Knowledge Base Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="rag-sub-header">Ingest foundational specifications and standard guidelines into the RAG engine.</div>', unsafe_allow_html=True)

    # Progressive state variables initialization
    if "extracted_chunks" not in st.session_state:
        st.session_state.extracted_chunks = None
    if "target_collection_name" not in st.session_state:
        st.session_state.target_collection_name = ""
    if "dialog_completed" not in st.session_state:
        st.session_state.dialog_completed = False
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "run_extraction" not in st.session_state:
        st.session_state.run_extraction = False
    if "trigger_extraction_phase" not in st.session_state:
        st.session_state.trigger_extraction_phase = False
    if "start_page" not in st.session_state:
        st.session_state.start_page = 1
    if "end_page" not in st.session_state:
        st.session_state.end_page = 1

    # File Uploader
    uploaded_file = st.file_uploader(
        "Drop foundational files here (.pdf, .txt, .csv, .md, .log, .xlsx, .xls)", 
        type=["pdf", "txt", "csv", "md", "log", "xlsx", "xls"],
        accept_multiple_files=False,
        key="rag_file_uploader"
    )

    if uploaded_file is None:
        # Reset state if cleared
        st.session_state.dialog_completed = False
        st.session_state.target_collection_name = ""
        st.session_state.extracted_chunks = None
        st.session_state.current_file = None
        st.session_state.run_extraction = False
        st.session_state.trigger_extraction_phase = False
    else:
        # Reset if different file uploaded
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.current_file = uploaded_file.name
            st.session_state.dialog_completed = False
            st.session_state.extracted_chunks = None
            st.session_state.target_collection_name = ""
            st.session_state.run_extraction = False
            st.session_state.trigger_extraction_phase = False
            st.rerun()

        # Handle the intermediate bridge rerun to close the dialog before heavy extraction
        if st.session_state.get("trigger_extraction_phase", False):
            st.session_state.trigger_extraction_phase = False
            st.session_state.run_extraction = True
            st.rerun()

        # Trigger dialog configuration if not completed
        if not st.session_state.dialog_completed:
            # Auto-open the dialog only when the file is newly uploaded
            if st.session_state.get("last_uploaded_file") != uploaded_file.name:
                st.session_state.last_uploaded_file = uploaded_file.name
                configure_target_collection_dialog(uploaded_file)
                
            st.warning("⚠️ Action Required: Configure collection settings and parameters inside the dialog.")
            if st.button("Open Settings Dialog", key="reopen_dlg_btn"):
                configure_target_collection_dialog(uploaded_file)

        # Extraction logic execution
        if st.session_state.get("run_extraction", False):
            st.markdown("---")
            st.markdown("### 🧱 Processing Document...")
            st.write(f"📍 **Target Collection:** `{st.session_state.target_collection_name}`")
            
            target_collection = st.session_state.target_collection_name
            collection_mode = st.session_state.get("collection_mode", "Create New Collection")
            
            # Recreate or setup collection metadata
            if collection_mode == "Create New Collection":
                st.session_state.rag.setup_collection(target_collection, recreate=True)
            else:
                st.session_state.rag.setup_collection(target_collection, recreate=False)

            if st.session_state.extracted_chunks is None:
                st.session_state.extracted_chunks = []

            is_pdf = uploaded_file.name.lower().endswith(".pdf")

            if is_pdf:
                # PDF workflow
                try:
                    pdf_bytes = uploaded_file.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"Error loading PDF: {e}")
                    st.session_state.run_extraction = False
                    st.stop()
                    
                # PyMuPDF uses 0-indexed pages
                pages_0_indexed = list(range(st.session_state.start_page - 1, st.session_state.end_page))
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                status_text.text("Analyzing layout and extracting Markdown tables...")
                
                # Single-pass layout analysis for requested pages
                try:
                    # Returns a list of dicts: [{"text": "markdown...", "metadata": {...}}, ...]
                    md_pages = pymupdf4llm.to_markdown(doc, pages=pages_0_indexed, page_chunks=True)
                except Exception as e:
                    st.error(f"Markdown layout extraction failed: {e}")
                    st.stop()

                completed = 0
                llm = st.session_state.llm

                def process_page(md_page_dict, actual_page_num):
                    text = md_page_dict.get("text", "")
                    chunks = generate_chunks_with_llm(text, actual_page_num, llm=llm)
                    for chunk in chunks:
                        chunk["id"] = str(uuid.uuid4())
                        chunk["status"] = "pending"
                    return chunks

                st.markdown("### 📥 Live Chunk Ingestion Preview")
                live_preview_container = st.container()

                MAX_WORKERS = 1
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Map the markdown text dictionary to the actual 1-indexed page number
                    futures = {
                        executor.submit(process_page, md_pages[i], pages_0_indexed[i] + 1): pages_0_indexed[i] + 1 
                        for i in range(len(md_pages))
                    }
                    
                    for future in as_completed(futures):
                        p = futures[future]
                        try:
                            page_chunks = future.result()
                            for chunk in page_chunks:
                                split_chunks = st.session_state.rag._split_oversized_chunk(chunk)
                                for c in split_chunks:
                                    try:
                                        st.session_state.rag.ingest_chunk(
                                            collection_name=target_collection,
                                            chunk_id=c["id"],
                                            text=c["text"],
                                            title=c.get("title", "Untitled"),
                                            metadata=c.get("metadata", {})
                                        )
                                        c["status"] = "ingested"
                                    except Exception as e:
                                        c["status"] = "error"
                                        st.error(f"❌ Failed to ingest chunk ({c.get('title', 'Untitled')}): {e}")
                                    
                                    st.session_state.extracted_chunks.append(c)
                                    with live_preview_container:
                                        render_single_chunk_card(c, len(st.session_state.extracted_chunks) - 1, is_live=True)
                                        st.markdown("<br>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error processing page {p}: {e}")
                        completed += 1
                        progress_bar.progress(completed / len(pages_0_indexed))
                        status_text.text(f"Chunked and Ingested {completed}/{len(pages_0_indexed)} pages...")

                status_text.text("Extraction & Ingestion completed successfully!")
            else:
                # Text files workflow
                try:
                    file_bytes = uploaded_file.read()
                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    st.session_state.run_extraction = False
                    st.stop()
                
                # Mock RAG process_file interface to get standard chunks locally
                temp_engine = st.session_state.rag
                # Clear standard doc list temporarily
                old_docs = temp_engine.documents
                temp_engine.documents = []
                temp_engine.process_file(uploaded_file.name, file_bytes, target_collection)
                
                st.markdown("### 📥 Live Chunk Ingestion Preview")
                live_preview_container = st.container()
                
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                total_docs = len(temp_engine.documents)
                
                for idx, doc in enumerate(temp_engine.documents):
                    chunk = {
                        "id": doc["id"],
                        "title": doc["title"],
                        "text": doc["text"],
                        "status": "pending",
                        "metadata": doc["metadata"]
                    }
                    
                    split_chunks = st.session_state.rag._split_oversized_chunk(chunk)
                    for c in split_chunks:
                        try:
                            st.session_state.rag.ingest_chunk(
                                collection_name=target_collection,
                                chunk_id=c["id"],
                                text=c["text"],
                                title=c.get("title", "Untitled"),
                                metadata=c.get("metadata", {})
                            )
                            c["status"] = "ingested"
                        except Exception as e:
                            c["status"] = "error"
                            st.error(f"❌ Failed to ingest chunk ({c.get('title', 'Untitled')}): {e}")
                        
                        st.session_state.extracted_chunks.append(c)
                        with live_preview_container:
                            render_single_chunk_card(c, len(st.session_state.extracted_chunks) - 1, is_live=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                    
                    progress_bar.progress((idx + 1) / total_docs)
                    status_text.text(f"Processed and Ingested chunk {idx+1}/{total_docs}...")
                
                # Restore engine documents
                temp_engine.documents = old_docs
                status_text.text("Extraction & Ingestion completed successfully!")

            st.session_state.run_extraction = False
            st.rerun()

    # Extracted Chunks Preview
    if st.session_state.extracted_chunks is not None and not st.session_state.get("run_extraction", False):
        st.markdown("---")
        st.markdown("### 📝 Extracted Chunks Preview")
        st.write(f"**Target Collection:** `{st.session_state.target_collection_name}`")
        
        chunks = st.session_state.extracted_chunks
        
        # Check matching ingestion status from memory
        for chunk in chunks:
            # Check if chunk ID already in locally loaded RAGEngine
            local_exists = any(doc.get("id") == chunk["id"] for doc in st.session_state.rag.documents)
            if local_exists:
                chunk["status"] = "ingested"

        pending_count = sum(1 for c in chunks if c["status"] == "pending")
        ingested_count = sum(1 for c in chunks if c["status"] == "ingested")
        
        st.write(f"**Total Chunks:** {len(chunks)} | **Pending:** {pending_count} | **Ingested:** {ingested_count}")
        
        col_bulk, col_clear = st.columns(2)
        with col_bulk:
            if pending_count > 0:
                if st.button("📥 Upload Chunks to vectorDB", type="primary", use_container_width=True, key="ingest_all_btn"):
                    with st.spinner("Embedding and uploading chunks..."):
                        try:
                            pending_list = [c for c in chunks if c["status"] == "pending"]
                            st.session_state.rag.ingest_chunks_batch(st.session_state.target_collection_name, pending_list)
                            
                            for c in pending_list:
                                c["status"] = "ingested"
                                
                            st.success("Successfully ingested all pending chunks!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Bulk ingestion failed: {e}")
            else:
                st.success("All chunks have been ingested successfully!")
                
        with col_clear:
            if st.button("🧹 Clear Extracted Chunks", use_container_width=True, key="clear_extracted_btn"):
                st.session_state.extracted_chunks = None
                st.session_state.target_collection_name = ""
                st.session_state.dialog_completed = False
                st.session_state.run_extraction = False
                st.session_state.trigger_extraction_phase = False
                st.rerun()
                
        st.markdown("---")
        
        # Expander for individual chunks
        with st.expander("Chunks Created", expanded=True):
            for idx, chunk in enumerate(chunks):
                render_single_chunk_card(chunk, idx, is_live=False)
                st.markdown("<br>", unsafe_allow_html=True)

    # Verification Tool
    st.markdown("---")
    with st.expander("🔍 Verify Collection Ingests (Run Query on Vector DB)"):
        # Select from database collections dynamically
        available_cols = st.session_state.rag.get_collections()
        selected_verify_col = st.selectbox("Select target collection to search:", available_cols)
        
        test_query = st.text_input("Enter verification search query:", placeholder="e.g. lane keep assist", key="test_query_in")
        limit = st.slider("Number of results to fetch:", min_value=1, max_value=5, value=2, key="test_limit_sld")
        
        if st.button("Search Vector DB", key="test_search_btn") and test_query:
            with st.spinner("Searching..."):
                try:
                    search_results = st.session_state.rag.search(
                        search_text=test_query,
                        collection_name=selected_verify_col,
                        top_k=limit
                    )
                    
                    if not search_results:
                        st.info("No matching chunks found in the selected collection.")
                    else:
                        for rank, r in enumerate(search_results, 1):
                            payload = r["payload"]
                            meta = payload.get("metadata", {})
                            t = payload.get("title", "Untitled")
                            txt = payload.get("text", "")
                            p = meta.get("page", "N/A")
                            itype = meta.get("item_type", "N/A")
                            iid = meta.get("item_id", "N/A")
                            score = r["score"]
                            
                            st.markdown(f"""
                            **Match #{rank} (Similarity Score: {score:.4f})**
                            * **Title:** `{t}` | **Page:** `{p}` | **Type:** `{itype}` | **ID:** `{iid}`
                            * **Text:** {txt}
                            ---
                            """)
                except Exception as e:
                    st.error(f"Search verification failed: {e}")


