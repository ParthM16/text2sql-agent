"""
Vector Store  FAISS-based Semantic Search
===========================================
Handles:
  1. Attribute correction  fixes user typos against known DB values
  2. Few-shot retrieval   finds the most relevant SQL examples
"""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import config


class VectorStore:
    """Manages FAISS vector indexes for few-shot retrieval and typo correction."""

    def __init__(self, model_name: str = None, encoder=None):
        model_name = model_name or config.EMBEDDING_MODEL_NAME
        if encoder is not None:
            self.encoder = encoder
        else:
            from src.retrieval.embeddings import get_embedding_model
            self.encoder = get_embedding_model(model_name)

        self.backend = "faiss"
        self.dimension = self.encoder.get_embedding_dimension()

        # Attribute correction index
        self.attribute_index = faiss.IndexFlatL2(self.dimension)
        self.attribute_mapping = []  # list of {'col': ..., 'val': ...}

        # Few-shot SQL example index
        self.few_shot_index = faiss.IndexFlatL2(self.dimension)
        self.few_shot_mapping = []  # list of {'query': ..., 'sql': ...}

        # Schema Table Index (for semantic table discovery & selection at 50+ table scale)
        self.table_index = faiss.IndexFlatL2(self.dimension)
        self.table_mapping = []  # list of {'table': str, 'summary': str}

        # Schema Column Index (for semantic column pruning / schema linking at 1,000+ column scale)
        self.column_index = faiss.IndexFlatL2(self.dimension)
        self.column_mapping = []  # list of {'table': str, 'column': str, 'type': str, 'description': str, 'is_pk': bool, 'is_fk': bool}

    # 
    # Attribute Correction
    # 
    def index_attributes(self, column: str, attributes: list):
        """Index known valid categorical values from DB.
        e.g. index_attributes('category', ['Electronics', 'Grocery'])
        """
        if not attributes:
            return
        embeddings = self.encoder.encode(attributes)
        self.attribute_index.add(np.array(embeddings).astype("float32"))
        for val in attributes:
            self.attribute_mapping.append({"col": column, "val": val})

    def match_attribute(self, query: str, k: int = 1):
        """Find the closest DB attribute matching the user's potential typo."""
        if self.attribute_index.ntotal == 0:
            return None
        query_emb = self.encoder.encode([query])
        _, indices = self.attribute_index.search(np.array(query_emb).astype("float32"), k)
        return self.attribute_mapping[indices[0][0]]

    # 
    # Few-Shot SQL Retrieval
    # 
    def add_few_shots(self, examples: list[dict]):
        """Add few-shot examples. Format: [{'query': str, 'sql': str}]"""
        if not examples:
            return
        queries = [ex["query"] for ex in examples]
        embeddings = self.encoder.encode(queries)
        self.few_shot_index.add(np.array(embeddings).astype("float32"))
        self.few_shot_mapping.extend(examples)

    def retrieve_few_shots(self, query: str, k: int = 3):
        """Retrieve the most relevant SQL examples using vector similarity."""
        if self.few_shot_index.ntotal == 0:
            return []
        k = min(k, self.few_shot_index.ntotal)
        query_emb = self.encoder.encode([query])
        _, indices = self.few_shot_index.search(np.array(query_emb).astype("float32"), k)
        return [self.few_shot_mapping[i] for i in indices[0]]

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
        embeddings = self.encoder.encode(summaries)
        self.table_index.add(np.array(embeddings).astype("float32"))
        self.table_mapping.extend(tables_data)

    def retrieve_relevant_tables(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieves top-k table names most semantically relevant to the user query."""
        if self.table_index.ntotal == 0:
            return []
        k = min(top_k, self.table_index.ntotal)
        query_emb = self.encoder.encode([query])
        _, indices = self.table_index.search(np.array(query_emb).astype("float32"), k)
        
        seen = set()
        results = []
        for i in indices[0]:
            if i < len(self.table_mapping):
                tbl = self.table_mapping[i]["table"]
                if tbl not in seen:
                    seen.add(tbl)
                    results.append(tbl)
        return results

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
        embeddings = self.encoder.encode(signatures)
        self.column_index.add(np.array(embeddings).astype("float32"))
        self.column_mapping.extend(columns_data)

    def retrieve_relevant_columns(self, query: str, candidate_tables: list[str] = None, top_k_per_table: int = 15) -> dict[str, list[str]]:
        """Retrieves top matching columns per table using vector similarity.
        Returns: {table_name: [col_name_1, col_name_2, ...]}
        """
        if self.column_index.ntotal == 0:
            return {}

        cand_set = {t.lower() for t in candidate_tables} if candidate_tables else None
        
        # Search a wider top-N in the global column index, then bucket by table
        fetch_k = min(max(50, (len(candidate_tables or [1]) * top_k_per_table * 3)), self.column_index.ntotal)
        query_emb = self.encoder.encode([query])
        _, indices = self.column_index.search(np.array(query_emb).astype("float32"), fetch_k)

        matched_cols = {}
        for i in indices[0]:
            if i < len(self.column_mapping):
                col_meta = self.column_mapping[i]
                tbl = col_meta["table"]
                tbl_lower = tbl.lower()
                if cand_set and tbl_lower not in cand_set:
                    continue
                
                col_list = matched_cols.setdefault(tbl, [])
                if len(col_list) < top_k_per_table and col_meta["column"] not in col_list:
                    col_list.append(col_meta["column"])

        return matched_cols

    def learn_from_correction(self, query, fixed_sql, error_msg, file_path):
        """
        [NEW] Automated Learning Loop. 
        Saves a successful correction as a few-shot example.
        """
        import json
        import os
        import numpy as np
        
        # 1. Fuzzy match against existing mapping using semantic distance
        is_duplicate = False
        duplicate_idx = -1
        
        if self.few_shot_index.ntotal > 0:
            query_emb = self.encoder.encode([query])
            D, I = self.few_shot_index.search(np.array(query_emb).astype("float32"), 1)
            # L2 distance threshold from centralized config (default < 0.3)
            if D[0][0] < config.DEDUP_DISTANCE_THRESHOLD: 
                 is_duplicate = True
                 duplicate_idx = I[0][0]
                 # Update in-memory mapping
                 self.few_shot_mapping[duplicate_idx]["sql"] = fixed_sql
                 self.few_shot_mapping[duplicate_idx]["note"] = f"Updated from correction: {error_msg}"
        
        # 2. Persistence to JSON
        try:
            examples = []
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    examples = json.load(f)
            
            if is_duplicate:
                target_q = self.few_shot_mapping[duplicate_idx]["query"].lower().strip()
                for ex in examples:
                    if ex["query"].lower().strip() == target_q:
                        ex["sql"] = fixed_sql
                        ex["note"] = f"Learned from error: {error_msg}"
                        break
            else:
                # Cap at 200 examples to ensure coverage without bloat
                if len(examples) < 200:
                    new_ex = {
                        "query": query,
                        "sql": fixed_sql,
                        "note": f"Learned from error: {error_msg}"
                    }
                    examples.append(new_ex)
                    # Re-index only the new one
                    self.add_few_shots([new_ex])
            
            # Save atomic-ly
            temp_path = file_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(examples, f, indent=2)
            os.replace(temp_path, file_path)

            # Dual-write sync to ChromaDB if active in memory
            try:
                from src.retrieval.store_factory import VectorStoreFactory
                VectorStoreFactory.sync_correction_to_counterpart(
                    source_store=self,
                    query=query,
                    fixed_sql=fixed_sql,
                    error_msg=error_msg,
                    is_duplicate=is_duplicate,
                    target_query=self.few_shot_mapping[duplicate_idx]["query"] if is_duplicate else None
                )
            except Exception as sync_err:
                print(f"[WARN] Dual-write sync to Chroma failed: {sync_err}")

            return True
        except Exception as e:
            print(f"[ERROR] Learning Loop failed: {e}")
            return False

