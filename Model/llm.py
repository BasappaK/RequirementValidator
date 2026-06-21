import os
from pathlib import Path
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Prioritize api_key.env fallback mapping
env_path = Path(".env") if Path(".env").exists() else Path("api_key.env")
load_dotenv(env_path)

class LLMManager:
    def __init__(self, model_name="gemini-2.5-flash-lite"):
        # Check Streamlit secrets first, fallback to environment variables
        gemini_api_key = None
        try:
            import streamlit as st
            gemini_api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("API_KEY")
        except Exception:
            pass
        if not gemini_api_key:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            
        nvidia_api_key = None
        try:
            import streamlit as st
            nvidia_api_key = st.secrets.get("NVIDIA_API_KEY") or st.secrets.get("API_KEY")
        except Exception:
            pass
        if not nvidia_api_key:
            nvidia_api_key = os.getenv("NVIDIA_API_KEY")

        # Fallback configurations
        if not gemini_api_key:
            gemini_api_key = nvidia_api_key
        if not nvidia_api_key:
            nvidia_api_key = gemini_api_key
           
        self.client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_api_key
        )
        self.embed_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        self.model_name = model_name
        self.embedding_model = "nvidia/nv-embedqa-e5-v5"
        self.retries = 3

    def _retry_api_call(self, api_func, *args, **kwargs):
        """Helper to execute API functions with transient failure retry logic using exponential backoff."""
        retries = getattr(self, "retries", 3)

        import time
        import re
        last_exception = None
        for attempt in range(retries + 1):
            try:
                # Proactively space out API calls to prevent hitting free-tier limits
                time.sleep(1.0)
                return api_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                err_msg = str(e)
                if attempt < retries:
                    # Check for rate limits / quota exceeded errors
                    if "429" in err_msg or "Too Many Requests" in err_msg or "rate limit" in err_msg.lower() or "quota" in err_msg.lower():
                        # Extract Google API's recommended retry delay dynamically
                        retry_match = re.search(r'retry\s+in\s+([\d\.]+)\s*s', err_msg, re.IGNORECASE)
                        if retry_match:
                            wait_time = float(retry_match.group(1)) + 1.5
                        else:
                            wait_time = (2 ** attempt) * 5
                        
                        status_msg = f"⏳ API Quota limit reached. Pausing & retrying in {wait_time:.1f}s... (Attempt {attempt+1}/{retries})"
                    else:
                        wait_time = 2 ** attempt
                        status_msg = f"⚠️ API call failed: {e}. Retrying in {wait_time}s... (Attempt {attempt+1}/{retries})"
                    
                    print(f"[LLMManager] {status_msg}", flush=True)
                    
                    # Log warning to the Streamlit UI dynamically
                    try:
                        import streamlit as st
                        st.toast(status_msg, icon="⏳")
                        st.warning(status_msg)
                    except Exception:
                        pass
                        
                    time.sleep(wait_time)
                else:
                    print(f"[LLMManager] API call failed after {retries} retries: {e}", flush=True)
                    raise last_exception

    def get_response(self, messages, stream=True):
        # Keep a sliding window of the last 10 messages to avoid token limit issues
        trimmed_messages = messages[-10:]
        
        def call_and_validate():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=trimmed_messages,
                temperature=0.0,
                top_p=0.01,
                max_tokens=8192,
                stream=stream
            )
            if not stream:
                if not response or not response.choices:
                    raise ValueError("LLM returned an empty response (no choices).")
                content = response.choices[0].message.content
                if content is None or content.strip() == "":
                    raise ValueError("LLM returned empty or null content.")
            return response

        return self._retry_api_call(call_and_validate)

    def get_embedding(self, text: str):
        """Generates contextual float embeddings using the NVIDIA NIM footprint"""
        response = self._retry_api_call(
            self.embed_client.embeddings.create,
            input=[text],
            model=self.embedding_model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str]):
        """Generates embeddings for a list of strings in a single batch API call."""
        if not texts:
            return []
        response = self._retry_api_call(
            self.embed_client.embeddings.create,
            input=texts,
            model=self.embedding_model,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "NONE"}
        )
        # Ensure correct ordering by sorting on the index property
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
