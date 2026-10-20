"""
ChromaDB Vector Store Adapter — Persistent Semantic Search
==========================================================
Provides persistent local vector storage using ChromaDB:
  1. Attribute correction — fixes user typos against known DB values
  2. Few-shot retrieval   — finds the most relevant SQL examples
  3. Automated learning   — upserts self-healed corrections with L2 deduplication

Reuses the shared SentenceTransformer singleton to prevent duplicate model loading in RAM.
"""
import os
import json
import hashlib
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings

from src.retrieval.embeddings import get_embedding_model
from src.config import config


class _IndexProxy:
    """Proxy providing FAISS-compatible `.ntotal` attribute for duck typing."""
    def __init__(self, collection):
        self._collection = collection

    @property
    def ntotal(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0


class ChromaVectorStore:
    """Manages persistent ChromaDB vector collections for few-shot retrieval and typo correction."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        model_name: Optional[str] = None,
        encoder: Optional[Any] = None,
    ):
        # Resolve persistence directory from centralized config
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_directory, exist_ok=True)

        # 🧠 Singleton embedding model — share existing instance across backends
        model_name = model_name or config.EMBEDDING_MODEL_NAME
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = get_embedding_model(model_name)

        self.backend = "chroma"
        self.dimension = self.encoder.get_embedding_dimension()

        # Initialize persistent ChromaDB client with telemetry disabled for fast, zero-delay local startup
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Collections with L2 distance space matching FAISS IndexFlatL2
        self.few_shot_collection = self.client.get_or_create_collection(
            name="few_shot_sql",
            metadata={"hnsw:space": "l2"}
        )
        self.attribute_collection = self.client.get_or_create_collection(
            name="categorical_attributes",
            metadata={"hnsw:space": "l2"}
        )
        self.table_collection = self.client.get_or_create_collection(
            name="schema_tables",
            metadata={"hnsw:space": "l2"}
        )
        self.column_collection = self.client.get_or_create_collection(
            name="schema_columns",
            metadata={"hnsw:space": "l2"}
        )

        # Duck-typing proxies for code expecting FAISS index.ntotal
        self.few_shot_index = _IndexProxy(self.few_shot_collection)
        self.attribute_index = _IndexProxy(self.attribute_collection)
        self.table_index = _IndexProxy(self.table_collection)
        self.column_index = _IndexProxy(self.column_collection)

        # In-memory mappings kept in sync for fast access
        self.attribute_mapping: List[Dict[str, str]] = []
        self.few_shot_mapping: List[Dict[str, Any]] = []
        self.table_mapping: List[Dict[str, str]] = []
        self.column_mapping: List[Dict[str, Any]] = []

        # Hydrate in-memory mapping from persistent collections if they already contain data
        self._hydrate_in_memory_mappings()

    def _hydrate_in_memory_mappings(self) -> None:
        """Hydrates few_shot_mapping and attribute_mapping from existing ChromaDB collections."""
        try:
            if self.few_shot_collection.count() > 0:
                data = self.few_shot_collection.get()
                for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
                    self.few_shot_mapping.append({
                        "query": doc,
                        "sql": meta.get("sql", ""),
                        "note": meta.get("note", "")
                    })
        except Exception as e:
            print(f"[WARN] Failed to hydrate few-shot mapping from Chroma: {e}")

        try:
            if self.attribute_collection.count() > 0:
                attr_data = self.attribute_collection.get()
                for doc, meta in zip(attr_data.get("documents", []), attr_data.get("metadatas", [])):
                    self.attribute_mapping.append({
                        "col": meta.get("col", ""),
                        "val": meta.get("val", doc)
                    })
        except Exception as e:
            print(f"[WARN] Failed to hydrate attribute mapping from Chroma: {e}")

    # ─────────────────────────────────────────────────────────────
    # Attribute Correction
    # ─────────────────────────────────────────────────────────────
    def index_attributes(self, column: str, attributes: list):
        """Index known valid categorical values from DB.
        e.g. index_attributes('category', ['Electronics', 'Grocery'])
        """
        if not attributes:
            return

        clean_attrs = [str(a).strip() for a in attributes if str(a).strip()]
        if not clean_attrs:
            return

        embeddings = self.encoder.encode(clean_attrs).tolist()
        ids = [f"{column}::{attr}" for attr in clean_attrs]
        metadatas = [{"col": column, "val": attr} for attr in clean_attrs]

        self.attribute_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=clean_attrs,
            metadatas=metadatas
        )

        for val in clean_attrs:
            entry = {"col": column, "val": val}
            if entry not in self.attribute_mapping:
                self.attribute_mapping.append(entry)

    def match_attribute(self, query: str, k: int = 1) -> Optional[Dict[str, str]]:
        """Find the closest DB attribute matching the user's potential typo."""
        if self.attribute_collection.count() == 0:
            return None

        query_emb = self.encoder.encode([query]).tolist()
        results = self.attribute_collection.query(
            query_embeddings=query_emb,
            n_results=min(k, self.attribute_collection.count())
        )

        if results and results.get("metadatas") and results["metadatas"][0]:
            top_meta = results["metadatas"][0][0]
            return {"col": top_meta.get("col", ""), "val": top_meta.get("val", "")}

        return None

    # ─────────────────────────────────────────────────────────────
    # Few-Shot SQL Retrieval
    # ─────────────────────────────────────────────────────────────
    def add_few_shots(self, examples: list[dict]):
        """Add few-shot examples. Format: [{'query': str, 'sql': str, (optional) 'note': str}]"""
        if not examples:
            return

        queries = [ex["query"] for ex in examples]
        embeddings = self.encoder.encode(queries).tolist()

        ids = [
            hashlib.sha256(ex["query"].strip().lower().encode("utf-8")).hexdigest()[:16]
            for ex in examples
        ]
        metadatas = [
            {"sql": ex["sql"], "note": ex.get("note", "")}
            for ex in examples
        ]

        self.few_shot_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=queries,
            metadatas=metadatas
        )

        existing_queries = {ex["query"].strip().lower() for ex in self.few_shot_mapping}
        for ex in examples:
            if ex["query"].strip().lower() not in existing_queries:
                self.few_shot_mapping.append(ex)
                existing_queries.add(ex["query"].strip().lower())

    def retrieve_few_shots(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve the most relevant SQL examples using vector similarity."""
        total_count = self.few_shot_collection.count()
        if total_count == 0:
            return []

        fetch_k = min(k, total_count)
        query_emb = self.encoder.encode([query]).tolist()
        results = self.few_shot_collection.query(
            query_embeddings=query_emb,
            n_results=fetch_k
        )

        few_shots = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            for doc, meta in zip(docs, metas):
                few_shots.append({
                    "query": doc,
                    "sql": meta.get("sql", ""),
                    "note": meta.get("note", "")
                })

        return few_shots

    # ─────────────────────────────────────────────────────────────
    # Enterprise Schema RAG: Table Catalog Semantic Index
    # ─────────────────────────────────────────────────────────────
    def index_tables(self, tables_data: list[dict]):
        """Indexes table summaries for semantic table retrieval (handles 50+ tables).
        Format: [{'table': 'customers', 'summary': 'Table: customers | Summary: ...'}]
        """
        if not tables_data:
            return

        summaries = [t["summary"] for t in tables_data]
        embeddings = self.encoder.encode(summaries).tolist()
        ids = [f"table::{t['table'].lower()}" for t in tables_data]
        metadatas = [{"table": t["table"]} for t in tables_data]

        self.table_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=summaries,
            metadatas=metadatas
        )

        existing_tables = {t["table"].lower() for t in self.table_mapping}
        for t in tables_data:
            if t["table"].lower() not in existing_tables:
                self.table_mapping.append(t)
                existing_tables.add(t["table"].lower())

    def retrieve_relevant_tables(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieves top-k table names most semantically relevant to the user query."""
        total_count = self.table_collection.count()
        if total_count == 0:
            return []

        fetch_k = min(top_k, total_count)
        query_emb = self.encoder.encode([query]).tolist()
        results = self.table_collection.query(
            query_embeddings=query_emb,
            n_results=fetch_k
        )

        seen = set()
        matched_tables = []
        if results and results.get("metadatas") and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                tbl = meta.get("table", "")
                if tbl and tbl not in seen:
                    seen.add(tbl)
                    matched_tables.append(tbl)

        return matched_tables

    # ─────────────────────────────────────────────────────────────
    # Enterprise Schema RAG: Column Pruning & Schema Linking Index
    # ─────────────────────────────────────────────────────────────
    def index_columns(self, columns_data: list[dict]):
        """Indexes column signatures for semantic schema linking / column pruning (handles 1,000+ columns).
        Format: [{'table': 'tbl', 'column': 'col', 'type': 'float', 'signature': 'Table: tbl, Column: col...'}]
        """
        if not columns_data:
            return

        signatures = [c["signature"] for c in columns_data]
        embeddings = self.encoder.encode(signatures).tolist()
        ids = [f"col::{c['table'].lower()}::{c['column'].lower()}" for c in columns_data]
        metadatas = [
            {
                "table": c["table"],
                "column": c["column"],
                "type": c.get("type", ""),
                "is_pk": str(c.get("is_pk", False)),
                "is_fk": str(c.get("is_fk", False))
            }
            for c in columns_data
        ]

        self.column_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=signatures,
            metadatas=metadatas
        )

        existing_cols = {(c["table"].lower(), c["column"].lower()) for c in self.column_mapping}
        for c in columns_data:
            key = (c["table"].lower(), c["column"].lower())
            if key not in existing_cols:
                self.column_mapping.append(c)
                existing_cols.add(key)

    def retrieve_relevant_columns(self, query: str, candidate_tables: list[str] = None, top_k_per_table: int = 15) -> dict[str, list[str]]:
        """Retrieves top matching columns per table using vector similarity.
        Returns: {table_name: [col_name_1, col_name_2, ...]}
        """
        total_count = self.column_collection.count()
        if total_count == 0:
            return {}

        cand_set = {t.lower() for t in candidate_tables} if candidate_tables else None
        fetch_k = min(max(50, (len(candidate_tables or [1]) * top_k_per_table * 3)), total_count)
        
        query_emb = self.encoder.encode([query]).tolist()
        results = self.column_collection.query(
            query_embeddings=query_emb,
            n_results=fetch_k
        )

        matched_cols = {}
        if results and results.get("metadatas") and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                tbl = meta.get("table", "")
                col = meta.get("column", "")
                if cand_set and tbl.lower() not in cand_set:
                    continue
                
                col_list = matched_cols.setdefault(tbl, [])
                if len(col_list) < top_k_per_table and col not in col_list:
                    col_list.append(col)

        return matched_cols
        """
        Automated Learning Loop with L2 Semantic Deduplication.
        Upserts the correction into ChromaDB and persists to JSON.
        """
        is_duplicate = False
        duplicate_id = None
        target_ex = None

        total_count = self.few_shot_collection.count()
        query_emb = self.encoder.encode([query]).tolist()

        if total_count > 0:
            results = self.few_shot_collection.query(
                query_embeddings=query_emb,
                n_results=1
            )
            # L2 distance threshold from centralized config (default < 0.3)
            if results and results.get("distances") and results["distances"][0]:
                dist = results["distances"][0][0]
                if dist < config.DEDUP_DISTANCE_THRESHOLD:
                    is_duplicate = True
                    duplicate_id = results["ids"][0][0]
                    doc_query = results["documents"][0][0]
                    target_ex = doc_query

        # Upsert into ChromaDB
        if is_duplicate and duplicate_id:
            self.few_shot_collection.update(
                ids=[duplicate_id],
                metadatas=[{"sql": fixed_sql, "note": f"Updated from correction: {error_msg}"}]
            )
            # Update in-memory mapping
            for ex in self.few_shot_mapping:
                if ex["query"].strip().lower() == target_ex.strip().lower():
                    ex["sql"] = fixed_sql
                    ex["note"] = f"Updated from correction: {error_msg}"
                    break
        else:
            new_id = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]
            new_meta = {"sql": fixed_sql, "note": f"Learned from error: {error_msg}"}
            self.few_shot_collection.upsert(
                ids=[new_id],
                embeddings=query_emb,
                documents=[query],
                metadatas=[new_meta]
            )
            self.few_shot_mapping.append({
                "query": query,
                "sql": fixed_sql,
                "note": f"Learned from error: {error_msg}"
            })

        # Persistence to JSON file
        try:
            examples = []
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    examples = json.load(f)

            if is_duplicate and target_ex:
                target_q = target_ex.lower().strip()
                for ex in examples:
                    if ex["query"].lower().strip() == target_q:
                        ex["sql"] = fixed_sql
                        ex["note"] = f"Learned from error: {error_msg}"
                        break
            else:
                if len(examples) < 200:
                    new_ex = {
                        "query": query,
                        "sql": fixed_sql,
                        "note": f"Learned from error: {error_msg}"
                    }
                    examples.append(new_ex)

            temp_path = file_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(examples, f, indent=2)
            os.replace(temp_path, file_path)

            # Dual-write sync to in-memory FAISS if active
            try:
                from src.retrieval.store_factory import VectorStoreFactory
                VectorStoreFactory.sync_correction_to_counterpart(
                    source_store=self,
                    query=query,
                    fixed_sql=fixed_sql,
                    error_msg=error_msg,
                    is_duplicate=is_duplicate,
                    target_query=target_ex if is_duplicate else None
                )
            except Exception as sync_err:
                print(f"[WARN] Dual-write sync to FAISS failed: {sync_err}")

            return True
        except Exception as e:
            print(f"[ERROR] ChromaDB Learning Loop JSON persistence failed: {e}")
            return False
