import os
from pathlib import Path
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Prioritize api_key.env fallback mapping
env_path = Path(".env") if Path(".env").exists() else Path("api_key.env")
load_dotenv(env_path)

class LLMManager:
    def __init__(self, model_name="gemini-2.5-flash"):
        # Check Streamlit secrets first, fallback to environment variable
        api_key = None
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("API_KEY")
        except Exception:
            pass
        
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("NVIDIA_API_KEY")
            
        self.client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key
        )
        self.model_name = model_name
        self.embedding_model = "gemini-embedding-001"
        self.retries = 3

    def _retry_api_call(self, api_func, *args, **kwargs):
        """Helper to execute API functions with transient failure retry logic using exponential backoff."""
        retries = getattr(self, "retries", 3)

        import time
        last_exception = None
        for attempt in range(retries + 1):
            try:
                return api_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < retries:
                    wait_time = 2 ** attempt
                    print(f"[LLMManager] API call failed: {e}. Retrying {attempt+1}/{retries} in {wait_time}s...", flush=True)
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
        """Generates contextual float embeddings using Google Gemini API"""
        response = self._retry_api_call(
            self.client.embeddings.create,
            input=[text],
            model=self.embedding_model,
            encoding_format="float"
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str]):
        """Generates embeddings for a list of strings in a single batch API call."""
        if not texts:
            return []
        response = self._retry_api_call(
            self.client.embeddings.create,
            input=texts,
            model=self.embedding_model,
            encoding_format="float"
        )
        # Ensure correct ordering by sorting on the index property
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
