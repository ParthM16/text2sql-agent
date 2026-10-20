"""
src/config.py - Centralized Application Configuration (Single Source of Truth)

All environment variables, model identifiers, API keys, fallback parameters,
vector store configurations, and pricing tables are defined here.
Never hardcode model names or filesystem paths in application code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project root directory (text2sql_agent/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env with override=True so changes on disk are immediately reflected
load_dotenv(PROJECT_ROOT / ".env", override=True)


class AppConfig:
    """Central configuration class providing typed, validated access to all app settings."""

    def __init__(self):
        self.reload()

    def reload(self):
        """Reloads .env from disk and re-harmonizes keys dynamically."""
        load_dotenv(PROJECT_ROOT / ".env", override=True)
        self._harmonize_api_keys()

    def _harmonize_api_keys(self):
        """
        Harmonizes GEMINI_API_KEY and GOOGLE_API_KEY in os.environ.
        If both are set, picks the active key and unsets GOOGLE_API_KEY from os.environ
        so the Google GenAI SDK never logs:
        'Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.'
        """
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        google_key = os.getenv("GOOGLE_API_KEY", "").strip()
        active_key = gemini_key or google_key
        if active_key == "your_api_key_here":
            active_key = ""

        if active_key:
            os.environ["GEMINI_API_KEY"] = active_key
            # Pop duplicate to silence SDK warning
            os.environ.pop("GOOGLE_API_KEY", None)

    # ── LLM Models & Providers ───────────────────────────────────
    @property
    def GEMINI_MODEL(self) -> str:
        """
        Default Gemini model name (default: 'gemini-3.8-flash').
        Override via GEMINI_MODEL in .env (e.g. 'gemini-3.8-flash', 'gemini-3.5-flash-lite').
        """
        return os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.8-flash"

    @property
    def OLLAMA_MODEL(self) -> str:
        """
        Default Ollama model name (default: 'qwen2.5-coder:3b').
        Override via OLLAMA_MODEL in .env (e.g. 'qwen2.5-coder:3b', 'llama3.2:3b').
        """
        return os.getenv("OLLAMA_MODEL", "").strip() or "qwen2.5-coder:3b"

    @property
    def OLLAMA_BASE_URL(self) -> str:
        """Base URL for local Ollama server."""
        return os.getenv("OLLAMA_BASE_URL", "").strip() or "http://localhost:11434"

    @property
    def DEFAULT_LLM_PROVIDER(self) -> str:
        """
        Primary LLM provider to use if none is specified by caller.
        Configurable via LLM_PROVIDER in .env. Defaults to GEMINI_MODEL.
        """
        env_provider = os.getenv("LLM_PROVIDER", "").strip()
        if env_provider:
            return env_provider
        return self.GEMINI_MODEL

    @property
    def MODEL_PRICING(self) -> dict:
        """
        Pricing lookup per 1M tokens (input / output) in USD.
        Dynamically adapts to current GEMINI_MODEL and OLLAMA_MODEL.
        """
        pricing = {
            self.GEMINI_MODEL: {"input_per_m": 0.075, "output_per_m": 0.30},
            "gemini-3.8-flash": {"input_per_m": 0.075, "output_per_m": 0.30},
            "gemini-3.5-flash-lite": {"input_per_m": 0.0375, "output_per_m": 0.15},
            "gemini-2.5-flash": {"input_per_m": 0.075, "output_per_m": 0.30},
            f"ollama/{self.OLLAMA_MODEL}": {"input_per_m": 0.0, "output_per_m": 0.0},
            "ollama/qwen2.5-coder:3b": {"input_per_m": 0.0, "output_per_m": 0.0},
            "gpt-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60},
            "claude-3.5-haiku": {"input_per_m": 0.80, "output_per_m": 4.00},
        }
        return pricing

    @property
    def UI_MODEL_OPTIONS(self) -> dict:
        """Dropdown options mapping human-readable label to provider string for Streamlit."""
        return {
            f"Gemini Cloud ({self.GEMINI_MODEL})": self.GEMINI_MODEL,
            f"Ollama Local ({self.OLLAMA_MODEL})": f"ollama/{self.OLLAMA_MODEL}",
        }

    # ── Vector Store & Retrieval ─────────────────────────────────
    @property
    def DEFAULT_VECTOR_BACKEND(self) -> str:
        """Default vector store backend ('faiss' or 'chroma'). Configurable via VECTOR_BACKEND in .env."""
        val = os.getenv("VECTOR_BACKEND", "").strip().lower()
        return val if val else "faiss"

    @property
    def EMBEDDING_MODEL_NAME(self) -> str:
        """SentenceTransformers embedding model identifier."""
        return os.getenv("EMBEDDING_MODEL", "").strip() or "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def CHROMA_PERSIST_DIR(self) -> str:
        """Directory path for persistent ChromaDB storage."""
        return os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))

    @property
    def FAISS_INDEX_DIR(self) -> str:
        """Directory path for FAISS index storage."""
        return os.getenv("FAISS_INDEX_DIR", str(PROJECT_ROOT / "data" / "faiss_index"))

    @property
    def FAISS_METADATA_PATH(self) -> str:
        """Metadata file path for FAISS."""
        return str(Path(self.FAISS_INDEX_DIR) / "metadata.json")

    @property
    def DEDUP_DISTANCE_THRESHOLD(self) -> float:
        """L2 distance threshold for vector deduplication in self-healing store."""
        try:
            return float(os.getenv("DEDUP_DISTANCE_THRESHOLD", "0.3"))
        except ValueError:
            return 0.3

    @property
    def UI_VECTOR_OPTIONS(self) -> dict:
        """Options mapping label to backend name for Streamlit UI."""
        return {
            "FAISS In-Memory (Fast L2)": "faiss",
            "ChromaDB Persistent (Local Disk)": "chroma",
        }

    # ── Enterprise Schema RAG Configuration ───────────────────────
    @property
    def SCHEMA_PRUNE_COLUMN_THRESHOLD(self) -> int:
        """Tables with more columns than this threshold undergo semantic column pruning."""
        try:
            return int(os.getenv("SCHEMA_PRUNE_COLUMN_THRESHOLD", "20"))
        except ValueError:
            return 20

    @property
    def SCHEMA_MAX_COLUMNS_PER_TABLE(self) -> int:
        """Maximum number of semantically retrieved columns to retain per wide table."""
        try:
            return int(os.getenv("SCHEMA_MAX_COLUMNS_PER_TABLE", "15"))
        except ValueError:
            return 15

    @property
    def TABLE_RETRIEVAL_THRESHOLD(self) -> int:
        """If database has more tables than this threshold, use vector search to pre-filter candidate tables."""
        try:
            return int(os.getenv("TABLE_RETRIEVAL_THRESHOLD", "6"))
        except ValueError:
            return 6

    @property
    def TABLE_RETRIEVAL_TOP_K(self) -> int:
        """Number of top candidate tables retrieved via vector similarity for large schemas."""
        try:
            return int(os.getenv("TABLE_RETRIEVAL_TOP_K", "6"))
        except ValueError:
            return 6

    # ── Credentials & GCP ────────────────────────────────────────
    @property
    def GEMINI_API_KEY(self) -> str:
        return os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def PROJECT_ID(self) -> str:
        return os.getenv("PROJECT_ID", "").strip()

    @property
    def LOCATION(self) -> str:
        return os.getenv("LOCATION", "us-central1").strip()

    @property
    def GOOGLE_APPLICATION_CREDENTIALS(self) -> str:
        return os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    # ── Database Credentials ─────────────────────────────────────
    @property
    def DB_DRIVER(self) -> str:
        return os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server").strip()

    @property
    def DB_SERVER_NAME(self) -> str:
        return os.getenv("DB_SERVER_NAME", r".\SQLEXPRESS").strip()

    @property
    def DB_DATABASE_NAME(self) -> str:
        return os.getenv("DB_DATABASE_NAME", "sales").strip()

    @property
    def DB_USER(self) -> str:
        return os.getenv("DB_USER", "").strip()

    @property
    def DB_PASSWORD(self) -> str:
        return os.getenv("DB_PASSWORD", "").strip()


# Singleton instance exported as single source of truth
config = AppConfig()
