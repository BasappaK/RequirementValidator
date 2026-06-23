import os
import numpy as np
import pickle
import uuid
import time
import textwrap
from pathlib import Path
from dotenv import load_dotenv
from Model.llm import LLMManager
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient



# Load environment variables
env_path = Path(".env") if Path(".env").exists() else Path("api_key.env")
load_dotenv(env_path)

QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()

class RAGEngine:
    def __init__(self, llm_manager: LLMManager, db_path="vector_store.pkl"):
        self.llm = llm_manager
        self.db_path = db_path
        self.documents = []  # List of dicts: {"id", "title", "text", "source", "collection", "metadata"}
        self.vectors = []    # List of embedding lists
        # --- NEW: Dynamically detect embedding dimension ---
        print("[RAGEngine] Detecting NVIDIA model embedding dimension...", flush=True)
        try:
            # Generate a test embedding to see how big it is
            dummy_vector = self.llm.get_embedding("test")
            self.embed_dim = len(dummy_vector)
            print(f"[RAGEngine] Successfully detected dimension: {self.embed_dim}", flush=True)
        except Exception as e:
            # Fallback just in case the API is temporarily unreachable during startup
            print(f"[RAGEngine] Failed to detect dimension. Defaulting to 1024. Error: {e}", flush=True)
            self.embed_dim = 1024


        # Initialize Qdrant Client if variables are set
        qdrant_url = os.environ.get("QDRANT_URL", "").strip()
        qdrant_api_key = os.environ.get("QDRANT_API_KEY", "").strip()
        
        try:
            import streamlit as st
            if "QDRANT_URL" in st.secrets:
                qdrant_url = st.secrets["QDRANT_URL"].strip()
            if "QDRANT_API_KEY" in st.secrets:
                qdrant_api_key = st.secrets["QDRANT_API_KEY"].strip()
        except Exception:
            pass

        self.qdrant_client = None
        if qdrant_url:
            try:
                from qdrant_client import QdrantClient
                self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
                print(f"[RAGEngine] QdrantClient connected to URL: {qdrant_url}", flush=True)
            except Exception as e:
                print(f"[RAGEngine] Failed to initialize QdrantClient: {e}", flush=True)

    def _safe_get_embedding(self, text: str, max_retries: int = 5) -> list:
        """Helper to get an embedding with exponential backoff on rate limits."""
        for attempt in range(max_retries):
            try:
                return self.llm.get_embedding(text)
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Too Many Requests" in err_msg or "rate limit" in err_msg.lower():
                    wait_time = 2 ** attempt
                    print(f"[RAGEngine] Rate limit hit. Retrying get_embedding in {wait_time}s... (Attempt {attempt+1}/{max_retries})", flush=True)
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception("Max retries exceeded for embeddings.")

    def _safe_get_embeddings_batch(self, texts: list[str], max_retries: int = 5) -> list:
        """Helper to get batch embeddings with exponential backoff on rate limits."""
        for attempt in range(max_retries):
            try:
                return self.llm.get_embeddings_batch(texts)
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Too Many Requests" in err_msg or "rate limit" in err_msg.lower():
                    wait_time = 2 ** attempt
                    print(f"[RAGEngine] Rate limit hit. Retrying get_embeddings_batch in {wait_time}s... (Attempt {attempt+1}/{max_retries})", flush=True)
                    time.sleep(wait_time)
                else:
                    raise e
        raise Exception("Max retries exceeded for batch embeddings.")

    def _split_oversized_chunk(self, chunk: dict, max_chars: int = 1800) -> list[dict]:
        """Splits a single chunk into multiple smaller chunks if it exceeds max_chars."""
        text = chunk.get("text", "")
        if len(text) <= max_chars:
            return [chunk]
        
        sub_chunks = []
        # Wrap text cleanly without cutting words in half
        text_pieces = textwrap.wrap(text, width=max_chars, break_long_words=False)
        
        original_title = chunk.get("title", "Untitled")
        
        for i, piece in enumerate(text_pieces):
            # Deep copy to prevent mutating the original metadata across iterations
            import copy
            new_chunk = copy.deepcopy(chunk)
            
            # FIXED: Generate a brand new, valid UUID for Qdrant
            new_chunk["id"] = str(uuid.uuid4())
            
            # Prepend context to subsequent chunks so context is not lost during retrieval
            if i > 0:
                new_chunk["text"] = f"[Continued from '{original_title}':] {piece}"
            else:
                new_chunk["text"] = piece
            
            # Store the original ID and part number in metadata for your reference
            if "metadata" not in new_chunk:
                new_chunk["metadata"] = {}
            new_chunk["metadata"]["parent_chunk_id"] = chunk.get("id")
            new_chunk["metadata"]["chunk_part"] = i + 1
            
            # Add a part identifier to the title for clarity
            new_chunk["title"] = f"{original_title} (Part {i+1})"
            
            sub_chunks.append(new_chunk)
            
        return sub_chunks

    def get_collections(self) -> list[str]:
        """List all available collections/knowledge bases."""
        if self.qdrant_client:
            try:
                return [c.name for c in self.qdrant_client.get_collections().collections]
            except Exception as e:
                print(f"[RAGEngine] Error fetching Qdrant collections: {e}", flush=True)
                return []
        else:
            # Fallback to local Pickle documents
            collections = set()
            for doc in self.documents:
                col = doc.get("collection")
                if col:
                    collections.add(col)
            return sorted(list(collections))

    def setup_collection(self, collection_name: str, recreate: bool = False):
        """Setup a collection. Create it in Qdrant, or clean it locally."""
        if self.qdrant_client:
            from qdrant_client.models import VectorParams, Distance, PayloadSchemaType, TextIndexParams, TokenizerType
            try:
                existing = [c.name for c in self.qdrant_client.get_collections().collections]
            except Exception:
                existing = []

            if collection_name in existing:
                if recreate:
                    try:
                        self.qdrant_client.delete_collection(collection_name)
                    except Exception as e:
                        print(f"[RAGEngine] Failed to delete collection {collection_name}: {e}", flush=True)
                else:
                    return

            try:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embed_dim,
                        distance=Distance.COSINE,
                        on_disk=False,
                    ),
                )
                # Create Payload keyword indexes
                for field in ["item_type", "item_id", "page"]:
                    try:
                        self.qdrant_client.create_payload_index(
                            collection_name=collection_name,
                            field_name=f"metadata.{field}",
                            field_schema=PayloadSchemaType.KEYWORD,
                        )
                    except Exception:
                        pass
                # Create text search index
                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="text",
                        field_schema=TextIndexParams(
                            type="text",
                            tokenizer=TokenizerType.WORD,
                            min_token_len=2,
                            max_token_len=40,
                            lowercase=True,
                        ),
                    )
                except Exception:
                    pass
            except Exception as e:
                print(f"[RAGEngine] Failed to create Qdrant collection: {e}", flush=True)
        else:
            # Local collection setup
            if recreate:
                indices_to_keep = [i for i, doc in enumerate(self.documents) if doc.get("collection") != collection_name]
                self.documents = [self.documents[i] for i in indices_to_keep]
                self.vectors = [self.vectors[i] for i in indices_to_keep]
                self._save_local_db()

    def clear_database(self, name: str):
        """Delete a collection from Qdrant."""
        if self.qdrant_client:
            try:
                self.qdrant_client.delete_collection(collection_name=name)
                print(f"[RAGEngine] Deleted collection {name} from Qdrant.", flush=True)
            except Exception as e:
                print(f"[RAGEngine] Failed to delete collection {name} from Qdrant: {e}", flush=True)

    def process_file(self, file_name, file_content, collection_name=None):
        """Decode and load file content blocks into memory for training."""
        if not collection_name:
            raise ValueError("collection_name is required when processing a file.")

        # 1. Handle Excel files directly from bytes
        if file_name.lower().endswith(('.xlsx', '.xls')):
            import io
            try:
                import pandas as pd
            except ImportError:
                raise ImportError("Missing dependencies for Excel. Please run: pip install pandas openpyxl")
            
            try:
                # Read the binary content into a pandas DataFrame
                df = pd.read_excel(io.BytesIO(file_content))
                df = df.fillna("")  # Replace NaNs with empty strings
                
                # --- THE FIX: BATCH ROWS INTO MARKDOWN TABLES ---
                chunk_size = 15 # Group 15 rows into a single context chunk
                
                for i in range(0, len(df), chunk_size):
                    chunk_df = df.iloc[i:i+chunk_size]
                    
                    # Manually construct a Markdown table (no extra dependencies needed)
                    headers = "| " + " | ".join(str(col) for col in chunk_df.columns) + " |"
                    separator = "| " + " | ".join(["---"] * len(chunk_df.columns)) + " |"
                    
                    md_rows = []
                    for _, row in chunk_df.iterrows():
                        md_rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
                        
                    markdown_table = "\n".join([headers, separator] + md_rows)
                    
                    # Add descriptive context to the top of the chunk
                    text_content = f"Document: {file_name}\nData Range: Rows {i+1} to {i+len(chunk_df)}\n\n{markdown_table}"
                    
                    self.documents.append({
                        "id": str(uuid.uuid4()),
                        "title": f"{file_name} (Rows {i+1}-{i+len(chunk_df)})",
                        "text": text_content,
                        "source": file_name,
                        "collection": collection_name,
                        "metadata": {
                            "item_type": "table_batch", 
                            "start_row": i+1, 
                            "end_row": i+len(chunk_df)
                        }
                    })
                return 
            except Exception as e:
                print(f"[RAGEngine] Failed to process Excel file {file_name}: {e}", flush=True)
                return

        # 2. Handle Text / CSV files
        try:
            text = file_content.decode("utf-8", errors="ignore")
        except Exception:
            text = str(file_content)

        if file_name.lower().endswith(".csv"):
            import csv
            from io import StringIO
            f = StringIO(text)
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                row_str = ", ".join(f"{k}: {v}" for k, v in row.items() if v)
                if row_str:
                    self.documents.append({
                        "id": str(uuid.uuid4()),
                        "title": f"Row {i+1}",
                        "text": row_str,
                        "source": f"{file_name} (Row {i+1})",
                        "collection": collection_name,
                        "metadata": {"item_type": "row", "page": 1, "item_id": f"R{i+1}"}
                    })
        else:
            # Segment by blank lines or paragraphs
            blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
            for i, block in enumerate(blocks):
                self.documents.append({
                    "id": str(uuid.uuid4()),
                    "title": f"Block {i+1}",
                    "text": block,
                    "source": f"{file_name} (Block {i+1})",
                    "collection": collection_name,
                    "metadata": {"item_type": "block", "page": 1}
                })

    def train_engine(self, progress_callback, collection_name=None):
        """Train engine: compute embeddings for loaded documents and save locally."""
        if not collection_name:
            raise ValueError("collection_name is required for training.")

        # Filter documents that belong to this training cycle
        new_docs = [doc for doc in self.documents if doc.get("collection") == collection_name and doc.get("id") not in [d.get("id") for d in self.documents if d in self.vectors]]
        if not self.documents:
            return False, "No documents loaded to train on."
        
        # Batch upload to Qdrant or compute locally
        total = len(self.documents)
        self.vectors = [None] * total
        
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_doc(idx, text):
            return idx, self._safe_get_embedding(text)

        completed = 0
        with ThreadPoolExecutor(max_workers=2  ) as executor:
            futures = {executor.submit(process_doc, idx, doc["text"]): idx for idx, doc in enumerate(self.documents)}
            for future in as_completed(futures):
                idx, embedding = future.result()
                self.vectors[idx] = embedding
                completed += 1
                if progress_callback:
                    progress_callback(completed / total)
        
        self._save_local_db()

        # If Qdrant is active, mirror the trained local database directly
        if self.qdrant_client:
            try:
                self.setup_collection(collection_name, recreate=True)
                self.ingest_chunks_batch(collection_name, self.documents)
            except Exception as e:
                print(f"[RAGEngine] Failed to sync training to Qdrant: {e}", flush=True)
            
        return True, f"Successfully trained and indexed {total} knowledge blocks!"

    def load_trained_engine(self) -> bool:
        """Load engine: sync from Qdrant if active."""
        if self.qdrant_client:
            try:
                collections = [c.name for c in self.qdrant_client.get_collections().collections]
                loaded_docs = []
                loaded_vecs = []
                
                for col in collections:
                    # Scroll through all points in this collection
                    offset = None
                    while True:
                        records, offset = self.qdrant_client.scroll(
                            collection_name=col,
                            limit=100,
                            with_payload=True,
                            with_vectors=True,
                            offset=offset
                        )
                        for r in records:
                            # Reconstruct doc_data from payload
                            payload = r.payload or {}
                            doc_data = {
                                "id": r.id,
                                "title": payload.get("title", "Untitled"),
                                "text": payload.get("text", ""),
                                "source": payload.get("source") or f"Page {payload.get('metadata', {}).get('page', 'N/A')}",
                                "collection": col,
                                "metadata": payload.get("metadata", {})
                            }
                            # Avoid duplicates by ID
                            if not any(d["id"] == r.id for d in loaded_docs):
                                loaded_docs.append(doc_data)
                                loaded_vecs.append(r.vector)
                        if offset is None:
                            break
                
                self.documents = loaded_docs
                self.vectors = loaded_vecs
                print(f"[RAGEngine] Successfully synced {len(self.documents)} documents from Qdrant collections.", flush=True)
                return True
            except Exception as e:
                print(f"[RAGEngine] Failed to sync from Qdrant: {e}.", flush=True)
        return False

    def _save_local_db(self):
        """No-op for Qdrant-only architecture. Preserves old pickle file."""
        pass

    def ingest_chunk(self, collection_name: str, chunk_id: str, text: str, title: str, metadata: dict):
        """Upsert a single document chunk (handles oversized chunks automatically)."""
        if metadata is None:
            metadata = {}
        metadata["embedding_model"] = self.llm.embedding_model

        # Package into a temporary dictionary to use our splitter
        temp_chunk = {
            "id": chunk_id,
            "text": text,
            "title": title,
            "metadata": metadata
        }
        
        # Sub-divide if necessary (returns a list of 1 or more chunks)
        processed_chunks = self._split_oversized_chunk(temp_chunk)

        for p_chunk in processed_chunks:
            p_id = p_chunk["id"]
            p_text = p_chunk["text"]
            p_title = p_chunk["title"]
            
            embedding = self._safe_get_embedding(p_text)

            doc_data = {
                "id": p_id,
                "title": p_title,
                "text": p_text,
                "source": p_chunk.get("source") or f"Page {p_chunk['metadata'].get('page', 'N/A')}",
                "collection": collection_name,
                "metadata": p_chunk["metadata"]
            }

            if self.qdrant_client:
                try:
                    point = PointStruct(
                        id=p_id,
                        vector=embedding,
                        payload={
                            "title": doc_data["title"],
                            "text": doc_data["text"],
                            "source": doc_data["source"],
                            "collection": doc_data["collection"],
                            "metadata": doc_data["metadata"]
                        }
                    )
                    self.qdrant_client.upsert(collection_name=collection_name, points=[point])
                except Exception as e:
                    print(f"[RAGEngine] Qdrant ingest failed: {e}. Ingesting locally.", flush=True)

            # Always ingest locally for backup & fallback
            existing_idx = None
            for idx, doc in enumerate(self.documents):
                if doc.get("id") == p_id:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                self.documents[existing_idx] = doc_data
                self.vectors[existing_idx] = embedding
            else:
                self.documents.append(doc_data)
                self.vectors.append(embedding)
                
        self._save_local_db()

    def ingest_chunks_batch(self, collection_name: str, chunks: list):
        """Batch upsert multiple document chunks (handles oversized chunks automatically)."""
        if not chunks:
            return

        # Ensure each chunk's metadata contains the embedding model name
        for chunk in chunks:
            if "metadata" not in chunk or chunk["metadata"] is None:
                chunk["metadata"] = {}
            chunk["metadata"]["embedding_model"] = self.llm.embedding_model

        # Flatten and split any oversized chunks before processing
        processed_chunks = []
        for chunk in chunks:
            processed_chunks.extend(self._split_oversized_chunk(chunk))

        texts = [c["text"] for c in processed_chunks]
        embeddings = self._safe_get_embeddings_batch(texts)

        points_to_upsert = []
        # Always save locally
        for chunk, emb in zip(processed_chunks, embeddings):
            chunk_id = chunk["id"]
            existing_idx = None
            for idx, doc in enumerate(self.documents):
                if doc.get("id") == chunk_id:
                    existing_idx = idx
                    break

            doc_data = {
                "id": chunk_id,
                "title": chunk.get("title", "Untitled"),
                "text": chunk["text"],
                "source": chunk.get("source") or f"Page {chunk.get('metadata', {}).get('page', 'N/A')}",
                "collection": collection_name,
                "metadata": chunk.get("metadata", {})
            }

            if existing_idx is not None:
                self.documents[existing_idx] = doc_data
                self.vectors[existing_idx] = emb
            else:
                self.documents.append(doc_data)
                self.vectors.append(emb)

            points_to_upsert.append(
                PointStruct(
                    id=chunk_id,
                    vector=emb,
                    payload={
                        "title": doc_data["title"],
                        "text": doc_data["text"],
                        "source": doc_data["source"],
                        "collection": doc_data["collection"],
                        "metadata": doc_data["metadata"]
                    }
                )
            )

        if self.qdrant_client:
            try:
                self.qdrant_client.upsert(collection_name=collection_name, points=points_to_upsert)
            except Exception as e:
                print(f"[RAGEngine] Qdrant batch ingest failed: {e}. Ingesting locally.", flush=True)

        self._save_local_db()

    def search(self, search_text: str, collection_name=None, top_k: int = 3) -> list[dict]:
        """Perform similarity search and return detailed list of results with scores."""
        if collection_name == "None":
            return []

        if self.qdrant_client:
            try:
                query_vector = self._safe_get_embedding(search_text)
                
                collections_to_search = []
                if not collection_name or collection_name == "All Collections":
                    collections_to_search = self.get_collections()
                elif isinstance(collection_name, list):
                    collections_to_search = collection_name
                else:
                    collections_to_search = [collection_name]

                all_points = []
                for col in collections_to_search:
                    try:
                        res = self.qdrant_client.query_points(
                            collection_name=col,
                            query=query_vector,
                            limit=top_k,
                            with_payload=True
                        )
                        for point in res.points:
                            all_points.append({
                                "id": point.id,
                                "score": point.score,
                                "payload": point.payload,
                                "collection": col
                            })
                    except Exception:
                        pass
                
                all_points.sort(key=lambda x: x["score"], reverse=True)
                return all_points[:top_k]
            except Exception as e:
                print(f"[RAGEngine] Qdrant search failed: {e}. Falling back to local search.", flush=True)

        # Local fallback
        if not self.vectors or len(self.vectors) == 0:
            return []

        query_vector = np.array(self._safe_get_embedding(search_text))
        matrix = np.array(self.vectors)

        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            return []

        scores = np.dot(matrix, query_vector) / (matrix_norms * query_norm + 1e-8)

        # Filter by collection
        valid_indices = []
        for idx, doc in enumerate(self.documents):
            doc_col = doc.get("collection", "")
            if not collection_name or collection_name == "All Collections":
                valid_indices.append(idx)
            elif isinstance(collection_name, list) and doc_col in collection_name:
                valid_indices.append(idx)
            elif not isinstance(collection_name, list) and doc_col == collection_name:
                valid_indices.append(idx)

        if not valid_indices:
            return []

        sorted_valid = sorted(valid_indices, key=lambda idx: scores[idx], reverse=True)
        top_indices = sorted_valid[:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "id": doc.get("id", str(uuid.uuid4())),
                "score": float(scores[idx]),
                "payload": {
                    "title": doc.get("title", "Untitled"),
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {})
                },
                "collection": doc.get("collection", "")
            })
        return results

    def query(self, search_text: str, collection_name=None, top_k: int = 3) -> str:
        """Query standard interface returning concatenated context blocks (backward compatible)."""
        results = self.search(search_text, collection_name, top_k)
        context_blocks = [r["payload"]["text"] for r in results if r["score"] > 0.3]
        return "\n\n--- Context Block ---\n".join(context_blocks)

    def query_batch(self, search_texts: list, collection_name=None, top_k: int = 3) -> list[str]:
        """Query standard interface for multiple search texts returning concatenated context blocks."""
        if not search_texts:
            return []

        results = []
        for q in search_texts:
            results.append(self.query(q, collection_name, top_k))
        return results

    def get_all_chunks(self, collection_name: str, limit: int = 100) -> list[dict]:
        """Retrieve all chunks from a specific collection using Qdrant's scroll API."""
        if not collection_name or collection_name == "None":
            return []

        if self.qdrant_client:
            try:
                # Use scroll to get raw points without needing a search query
                records, next_page_offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False  # We only need the text/metadata for UI, not the arrays
                )
                
                results = []
                for record in records:
                    results.append({
                        "id": record.id,
                        "payload": record.payload,
                        "collection": collection_name
                    })
                return results
            except Exception as e:
                print(f"[RAGEngine] Qdrant scroll failed: {e}. Falling back to local.", flush=True)

        # Local Database Fallback
        results = []
        count = 0
        for doc in self.documents:
            if doc.get("collection") == collection_name:
                results.append({
                    "id": doc.get("id"),
                    "payload": {
                        "title": doc.get("title", "Untitled"),
                        "text": doc.get("text", ""),
                        "metadata": doc.get("metadata", {})
                    },
                    "collection": doc.get("collection")
                })
                count += 1
                if count >= limit:
                    break
        return results