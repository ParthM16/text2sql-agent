"""
Vector Store Factory — Unified Vector Backend Provider
======================================================
Provides a single entry point for instantiating vector stores:
  - "faiss": In-memory FAISS IndexFlatL2 store (src.retrieval.vector_store)
  - "chroma" / "chromadb": Persistent local ChromaDB store (src.retrieval.chroma_store)

Ensures the SentenceTransformer embedding model is loaded as a singleton
and shared across backends to prevent double RAM allocation (~90 MB savings).
Maintains dual-write synchronization for the self-learning correction loop.
"""
import os
import hashlib
from typing import Optional, Union, Any
from src.retrieval.embeddings import get_embedding_model, register_embedding_model
from src.config import config


class VectorStoreFactory:
    """Factory for creating and managing FAISS or ChromaDB vector store backends."""

    _faiss_instance: Optional[Any] = None
    _chroma_instance: Optional[Any] = None

    @classmethod
    def get_vector_store(
        cls,
        backend: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Returns the configured vector store backend.
        
        Args:
            backend: "faiss" or "chroma". If None, reads from config.DEFAULT_VECTOR_BACKEND.
            model_name: Embedding model name. If None, reads from config.EMBEDDING_MODEL_NAME.
            **kwargs: Extra parameters passed to the store constructor (e.g. persist_directory).
            
        Returns:
            VectorStore or ChromaVectorStore instance.
        """
        # Lazy imports to prevent circular dependencies
        from src.retrieval.vector_store import VectorStore
        from src.retrieval.chroma_store import ChromaVectorStore

        backend = (backend or config.DEFAULT_VECTOR_BACKEND).strip().lower()
        model_name = model_name or config.EMBEDDING_MODEL_NAME

        if backend in ("chroma", "chromadb"):
            if cls._chroma_instance is None:
                shared_encoder = (
                    cls._faiss_instance.encoder
                    if cls._faiss_instance is not None
                    else get_embedding_model(model_name)
                )
                cls._chroma_instance = ChromaVectorStore(
                    encoder=shared_encoder,
                    model_name=model_name,
                    **kwargs
                )
                cls._seed_if_empty(cls._chroma_instance)
            return cls._chroma_instance

        # Default to FAISS
        if cls._faiss_instance is None:
            shared_encoder = (
                cls._chroma_instance.encoder
                if cls._chroma_instance is not None
                else get_embedding_model(model_name)
            )
            cls._faiss_instance = VectorStore(
                model_name=model_name,
                encoder=shared_encoder
            )
            cls._seed_if_empty(cls._faiss_instance)

        return cls._faiss_instance

    @classmethod
    def index_database_schema(cls, store: Any, schema_data: dict) -> None:
        """Indexes table summaries and column signatures into the vector store.
        
        Enables:
          - Semantic table retrieval for 50+ table enterprise databases.
          - Semantic column pruning / schema linking for 1,000+ column wide tables.
        """
        if not schema_data or not isinstance(schema_data, dict):
            return

        tables = schema_data.get("tables", {})
        relationships = schema_data.get("relationships", [])

        # 1. Prepare table documents
        table_docs = []
        for t_name, t_info in tables.items():
            desc = t_info.get("description", "").strip()
            pks = t_info.get("primary_keys", [])
            fks = [
                f"{r.get('from_column')} -> {r.get('to_table')}.{r.get('to_column')}"
                for r in relationships if r.get("from_table") == t_name
            ]
            summary = f"Table: {t_name}"
            if desc:
                summary += f" | Summary: {desc}"
            if pks:
                summary += f" | Primary Keys: [{', '.join(pks)}]"
            if fks:
                summary += f" | Foreign Keys: [{', '.join(fks)}]"
            table_docs.append({"table": t_name, "summary": summary})

        if hasattr(store, "index_tables") and table_docs:
            store.index_tables(table_docs)

        # 2. Prepare column documents
        col_docs = []
        for t_name, t_info in tables.items():
            pks = set(t_info.get("primary_keys", []))
            for c_name, c_info in t_info.get("columns", {}).items():
                c_type = c_info.get("type", "")
                c_desc = c_info.get("description", "").strip()
                is_pk = c_info.get("is_pk", c_name in pks)
                is_fk = c_info.get("is_fk", False)

                sig_parts = [f"Table: {t_name}", f"Column: {c_name}"]
                if c_type:
                    sig_parts.append(f"Type: {c_type}")
                if is_pk:
                    sig_parts.append("Primary Key")
                if is_fk:
                    sig_parts.append("Foreign Key")
                if c_desc:
                    sig_parts.append(f"Description: {c_desc}")

                signature = ", ".join(sig_parts)
                col_docs.append({
                    "table": t_name,
                    "column": c_name,
                    "type": c_type,
                    "description": c_desc,
                    "is_pk": is_pk,
                    "is_fk": is_fk,
                    "signature": signature
                })

        if hasattr(store, "index_columns") and col_docs:
            store.index_columns(col_docs)

    @classmethod
    def _seed_if_empty(cls, store: Any) -> None:
        """Seeds the store with few-shot examples and schema vectors if empty."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 1. Seed few-shot SQL examples
        try:
            if hasattr(store, "few_shot_index") and getattr(store.few_shot_index, "ntotal", 0) == 0:
                seed_path = os.path.join(base_dir, "data", "few_shot_examples.json")
                if os.path.exists(seed_path):
                    import json
                    with open(seed_path, "r", encoding="utf-8") as f:
                        seeds = json.load(f)
                    store.add_few_shots(seeds)
        except Exception:
            pass

        # 2. Seed schema tables & columns index
        try:
            if hasattr(store, "table_index") and getattr(store.table_index, "ntotal", 0) == 0:
                schema_path = os.path.join(base_dir, "data", "schema_knowledge.json")
                if os.path.exists(schema_path):
                    import json
                    with open(schema_path, "r", encoding="utf-8") as f:
                        schema_data = json.load(f)
                    cls.index_database_schema(store, schema_data)
        except Exception:
            pass

    @classmethod
    def sync_correction_to_counterpart(
        cls,
        source_store: Any,
        query: str,
        fixed_sql: str,
        error_msg: str,
        is_duplicate: bool,
        target_query: Optional[str] = None
    ) -> None:
        """Dual-writes a learned correction to the counterpart store if both are active in memory."""
        from src.retrieval.chroma_store import ChromaVectorStore

        # 1. Source was Chroma -> sync into FAISS in-memory if initialized
        if isinstance(source_store, ChromaVectorStore):
            if cls._faiss_instance is not None:
                try:
                    match_q = (target_query or query).strip().lower()
                    if is_duplicate:
                        for ex in cls._faiss_instance.few_shot_mapping:
                            if ex.get("query", "").strip().lower() == match_q:
                                ex["sql"] = fixed_sql
                                ex["note"] = f"Updated from correction: {error_msg}"
                                break
                    else:
                        new_ex = {
                            "query": query,
                            "sql": fixed_sql,
                            "note": f"Learned from error: {error_msg}"
                        }
                        cls._faiss_instance.add_few_shots([new_ex])
                except Exception as e:
                    print(f"[WARN] Failed to sync correction to in-memory FAISS: {e}")

        # 2. Source was FAISS -> sync into ChromaDB if initialized
        else:
            if cls._chroma_instance is not None:
                try:
                    query_emb = source_store.encoder.encode([query]).tolist()
                    if is_duplicate:
                        res = cls._chroma_instance.few_shot_collection.query(
                            query_embeddings=query_emb,
                            n_results=1
                        )
                        if res and res.get("ids") and res["ids"][0]:
                            dup_id = res["ids"][0][0]
                            cls._chroma_instance.few_shot_collection.update(
                                ids=[dup_id],
                                metadatas=[{"sql": fixed_sql, "note": f"Updated from correction: {error_msg}"}]
                            )
                    else:
                        new_id = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]
                        cls._chroma_instance.few_shot_collection.upsert(
                            ids=[new_id],
                            embeddings=query_emb,
                            documents=[query],
                            metadatas=[{"sql": fixed_sql, "note": f"Learned from error: {error_msg}"}]
                        )
                except Exception as e:
                    print(f"[WARN] Failed to sync correction to ChromaDB: {e}")

    @classmethod
    def reset(cls) -> None:
        """Resets cached store instances (useful in tests or reconfiguration)."""
        cls._faiss_instance = None
        cls._chroma_instance = None


def get_vector_store(backend: Optional[str] = None, **kwargs) -> Any:
    """Convenience function returning the active vector store backend."""
    return VectorStoreFactory.get_vector_store(backend=backend, **kwargs)
