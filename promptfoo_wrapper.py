"""
promptfoo_wrapper.py
Thin adapter so promptfoo can call our chatbot via:
  type: python
  id: "file://promptfoo_wrapper.py:predict"
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

_store = None
def _get_store():
    global _store
    if _store is None:
        from retrieve import load_store, DEFAULT_PERSIST_DIR, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
        _store = load_store(DEFAULT_PERSIST_DIR, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL)
    return _store

def predict(prompt: str, options: dict = None, context: dict = None) -> str:
    """Called by promptfoo for each test case."""
    try:
        from generate import generate_answer
        answer, _, _, _, _ = generate_answer(
            _get_store(), prompt, "mistral-small-latest", 5, None
        )
        return answer
    except Exception as e:
        return f"[ERROR: {e}]"
