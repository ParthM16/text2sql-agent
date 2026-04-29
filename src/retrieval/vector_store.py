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


class VectorStore:
    """Manages FAISS vector indexes for few-shot retrieval and typo correction."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        import os
        import socket
        import numpy as np

        print("\n" + "=" * 50)
        print(f"[INIT] Loading Vector AI Model: '{model_name}' into memory...")
        
        # PRE-EMPTIVE CONNECTIVITY CHECK (To avoid the 5-retry hanging delay)
        is_offline = False
        try:
            # Fast check (0.5s timeout) to see if we can reach HuggingFace
            socket.create_connection(("huggingface.co", 443), timeout=0.5)
        except (socket.timeout, OSError):
            is_offline = True
            print("[OFFLINE] Network unreachable. Forcing offline mode immediately...")
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'

        try:
            # Try loading (will be instant if is_offline is True)
            self.encoder = SentenceTransformer(model_name, local_files_only=is_offline)
        except Exception as e:
            # Final fallback if the first attempt failed for non-network reasons
            if not is_offline:
                print(f"[WARN] Status: Model verification failed ({type(e).__name__}). Switching to Local-Only...")
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                self.encoder = SentenceTransformer(model_name, local_files_only=True)
            else:
                print(f"[ERROR] Error: Model '{model_name}' not found locally. Please connect once to download.")
                raise e

        self.dimension = self.encoder.get_embedding_dimension()
        print(" Vector Model Loaded Successfully!")
        print("=" * 50 + "\n")

        # Attribute correction index
        self.attribute_index = faiss.IndexFlatL2(self.dimension)
        self.attribute_mapping = []  # list of {'col': ..., 'val': ...}

        # Few-shot SQL example index
        self.few_shot_index = faiss.IndexFlatL2(self.dimension)
        self.few_shot_mapping = []  # list of {'query': ..., 'sql': ...}

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
            # L2 distance < 0.3 on normalized vectors usually means > 85% semantic match
            if D[0][0] < 0.3: 
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
            return True
        except Exception as e:
            print(f"[ERROR] Learning Loop failed: {e}")
            return False

