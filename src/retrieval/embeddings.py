"""
Shared Singleton Embedding Model Manager
========================================
Ensures SentenceTransformer('all-MiniLM-L6-v2') is loaded exactly once
and shared across FAISS, ChromaDB, and other retrieval components.
"""
import os
import socket
from typing import Optional
from sentence_transformers import SentenceTransformer

# Automatically set HF_HOME to project's local models/ directory if not set
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_models_dir = os.path.join(base_dir, "models")
if os.path.exists(project_models_dir):
    os.environ.setdefault("HF_HOME", project_models_dir)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Process-level singleton dictionary keyed by model name
_EMBEDDING_SINGLETONS: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Returns a shared, cached singleton instance of the SentenceTransformer model.
    
    If the model is already loaded in memory, returns the existing instance immediately.
    Prioritizes instant local loading from disk, falling back to online download only if needed.
    """
    global _EMBEDDING_SINGLETONS

    if model_name in _EMBEDDING_SINGLETONS:
        return _EMBEDDING_SINGLETONS[model_name]

    print("\n" + "=" * 50)
    print(f"[INIT] Loading Vector AI Model: '{model_name}' into memory (Singleton)...")

    encoder = None

    # 1. Fast local attempt: Loads instantly from local cache if present
    try:
        encoder = SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        pass

    # 2. If not found locally, attempt network download with connectivity check
    if encoder is None:
        is_offline = False
        try:
            socket.create_connection(("huggingface.co", 443), timeout=0.5)
        except (socket.timeout, OSError):
            is_offline = True
            print("[OFFLINE] Network unreachable. Forcing offline mode...")
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'

        try:
            encoder = SentenceTransformer(model_name, local_files_only=is_offline)
        except Exception as e:
            if not is_offline:
                print(f"[WARN] Status: Model verification failed ({type(e).__name__}). Switching to Local-Only...")
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                encoder = SentenceTransformer(model_name, local_files_only=True)
            else:
                print(f"[ERROR] Error: Model '{model_name}' not found locally. Please connect once to download.")
                raise e

    print(" Vector Model Loaded Successfully! (Shared Singleton)")
    print("=" * 50 + "\n")

    _EMBEDDING_SINGLETONS[model_name] = encoder
    return encoder


def register_embedding_model(encoder: SentenceTransformer, model_name: str = "all-MiniLM-L6-v2") -> None:
    """Registers an already-instantiated SentenceTransformer instance as the singleton."""
    global _EMBEDDING_SINGLETONS
    _EMBEDDING_SINGLETONS[model_name] = encoder
