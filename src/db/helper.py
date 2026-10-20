"""
Database Helper  Connection & Query Abstraction Layer
=======================================================
Manages SQL Server connections (via pyodbc) and provides a
CSV/Excel  in-memory SQLite fallback for ad-hoc analysis.

Phase 1 additions:
  - SQL safety guard: blocks DDL/DML keywords before execution
  - Structured logging via src.utils.logger
  - DatabaseError raised with user-safe messages
  - File upload security: filename sanitization + extension whitelist
"""
import os
import re
import json
import urllib.parse
import pyodbc
import pandas as pd
import warnings

# Silence pandas SQLAlchemy warning for pure pyodbc connections
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()

logger = get_logger("db.helper")

KNOWLEDGE_PATH = os.path.join("data", "schema_knowledge.json")

#  SQL Safety Guard 
# These are only checked against the *generated* SQL  not the user's NL input.
BLOCKED_SQL_KEYWORDS = {
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER",
    "CREATE", "EXEC", "EXECUTE", "TRUNCATE", "GRANT", "REVOKE",
}

ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILENAME_LENGTH = 100
MAX_UPLOAD_SIZE_MB = 10


class DBHelper:
    """Unified database interface supporting SQL Server and in-memory SQLite."""

    def __init__(self):
        self.driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
        self.server = os.getenv("DB_SERVER_NAME", ".\\SQLEXPRESS")
        self.database = os.getenv("DB_DATABASE_NAME", "")
        self.user = os.getenv("DB_USER", "")
        self.password = os.getenv("DB_PASSWORD", "")

        self.in_memory_db = "sqlite:///:memory:"
        self.sqlite_engine = None
        self.sqlserver_engine = None
        self.mode = "sqlserver"  # 'sqlserver' | 'sqlite'

    # 
    # SQL Safety Guard
    # 
    def _validate_sql(self, sql: str) -> tuple[bool, str]:
        """Scans generated SQL for dangerous keywords. Returns (is_safe, reason)."""
        # Tokenize on whitespace and strip SQL comments
        cleaned = re.sub(r"--.*", "", sql)        # remove line comments
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)  # block comments
        tokens = set(re.findall(r"\b[A-Z]+\b", cleaned.upper()))
        blocked = tokens & BLOCKED_SQL_KEYWORDS
        if blocked:
            keyword = next(iter(blocked))
            return False, f"Blocked: generated SQL contains forbidden operation '{keyword}'"
        return True, ""

    # 
    # SQLAlchemy Engine Builder
    # 
    def _get_sqlserver_engine(self):
        """Lazily creates a SQLAlchemy engine for SQL Server."""
        if self.sqlserver_engine:
            return self.sqlserver_engine
            
        import urllib.parse
        if self.user and self.password:
            params = urllib.parse.quote_plus(
                f"Driver={{{self.driver}}};Server={self.server};"
                f"Database={self.database};UID={self.user};PWD={self.password};"
            )
        else:
            params = urllib.parse.quote_plus(
                f"Driver={{{self.driver}}};Server={self.server};"
                f"Database={self.database};Trusted_Connection=yes;"
            )
        conn_str = f"mssql+pyodbc:///?odbc_connect={params}"
        self.sqlserver_engine = create_engine(conn_str, pool_pre_ping=True)
        return self.sqlserver_engine

    # 
    # Query Execution
    # 
    def execute_query(self, query: str):
        """Executes a SQL query and returns (list[dict], error_or_None).
        
        Always validates SQL for dangerous operations before running.
        """
        is_safe, reason = self._validate_sql(query)
        if not is_safe:
            logger.warning(f"[SQL Safety) Blocked query: {reason} | SQL: {query[:200]}")
            return None, reason

        try:
            if self.mode == "sqlite" and self.sqlite_engine:
                engine = self.sqlite_engine
                mode_label = "SQLite"
            else:
                engine = self._get_sqlserver_engine()
                mode_label = "SQL Server"

            df = pd.read_sql(query, engine)
            
            # Deduplicate column names to prevent dictionary key collisions
            cols = pd.Series(df.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols[cols == dup].index.values.tolist()] = [
                    f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))
                ]
            df.columns = cols

            logger.info(f"[{mode_label}] Executed query. Rows returned: {len(df)}")
            return df.to_dict(orient="records"), None
            
        except Exception as e:
            logger.error(f"[SQL Exec] Unexpected error: {e}", exc_info=True)
            return None, str(e)

    # 
    # Knowledge Engine (Persistent Schema Cache)
    # 
    # ─────────────────────────────────────────────────────────────
    # Knowledge Engine (Persistent Schema Cache & Enterprise Catalog)
    # ─────────────────────────────────────────────────────────────
    def sync_knowledge_base(self):
        """Fetches latest schema (with PKs/FKs) and merges with existing descriptions in local JSON."""
        logger.info(f"[Knowledge] Syncing enterprise knowledge base for mode: {self.mode}...")
        
        # 1. Fetch live structured schema
        new_schema_data = self.get_schema_structured()
        if isinstance(new_schema_data, str) and "ERROR" in new_schema_data:
            return False, new_schema_data

        new_tables = new_schema_data.get("tables", {})
        live_relationships = new_schema_data.get("relationships", [])

        # 2. Load existing knowledge (to preserve user descriptions and custom notes)
        existing_knowledge = {}
        existing_relationships = []
        if os.path.exists(KNOWLEDGE_PATH):
            try:
                with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    existing_knowledge = cached_data.get("tables", {})
                    existing_relationships = cached_data.get("relationships", [])
            except Exception:
                logger.warning("[Knowledge] Could not load existing knowledge for merging.")

        # 3. Merge: Preserve existing descriptions for tables and columns
        for table_name, table_info in new_tables.items():
            if table_name in existing_knowledge:
                # Table exists? Keep table description
                cached_t = existing_knowledge[table_name]
                table_info["description"] = cached_t.get("description", "") or table_info.get("description", "")
                
                # Column check: Keep column descriptions
                existing_cols = cached_t.get("columns", {})
                for col_name, col_info in table_info.get("columns", {}).items():
                    if col_name in existing_cols:
                        col_info["description"] = existing_cols[col_name].get("description", "") or col_info.get("description", "")

        # Merge relationships: combine live and existing without duplicates
        rel_set = set()
        merged_relationships = []
        for r in live_relationships + existing_relationships:
            key = (r.get("from_table"), r.get("from_column"), r.get("to_table"), r.get("to_column"))
            if key not in rel_set and all(key):
                rel_set.add(key)
                merged_relationships.append(r)

        knowledge = {
            "mode": self.mode,
            "database": self.database if self.mode == "sqlserver" else "sqlite_memory",
            "tables": new_tables,
            "relationships": merged_relationships,
            "updated_at": pd.Timestamp.now().isoformat()
        }

        try:
            os.makedirs(os.path.dirname(KNOWLEDGE_PATH), exist_ok=True)
            with open(KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
                json.dump(knowledge, f, indent=4)
            logger.info(f"[Knowledge] Knowledge Base saved to {KNOWLEDGE_PATH}")
            return True, None
        except Exception as e:
            logger.error(f"[Knowledge] Failed to save knowledge base: {e}")
            return False, str(e)

    def get_knowledge_structured(self) -> dict:
        """Retrieves structured schema knowledge dictionary from local JSON."""
        if not os.path.exists(KNOWLEDGE_PATH):
            return {}
        try:
            with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                knowledge = json.load(f)
            if knowledge.get("mode") == self.mode:
                if self.mode == "sqlserver" and knowledge.get("database") != self.database:
                    return {}
                return knowledge
        except Exception as e:
            logger.warning(f"[Knowledge] Failed to parse knowledge base: {e}")
        return {}

    def get_table_catalog(self) -> str:
        """Returns a lean Table Catalog (names, business summaries, PKs/FKs).
        
        CRITICAL FOR ENTERPRISE SCALE (50+ tables):
        Does NOT dump all 1,000 columns into the Table Selection node.
        Only provides table signatures and high-level summaries so the LLM or vector search
        can pick relevant tables using < 500 tokens instead of 25,000 tokens.
        """
        knowledge = self.get_knowledge_structured()
        tables = knowledge.get("tables", {})
        if not tables:
            # Fallback to direct schema if cache is empty
            raw_struct = self.get_schema_structured()
            tables = raw_struct.get("tables", {}) if isinstance(raw_struct, dict) else {}

        relationships = knowledge.get("relationships", [])
        
        catalog_lines = []
        for table_name, info in tables.items():
            desc = info.get("description", "").strip()
            pks = info.get("primary_keys", [])
            
            # Find foreign keys involving this table
            fks = []
            for r in relationships:
                if r.get("from_table") == table_name:
                    fks.append(f"{r.get('from_column')} -> {r.get('to_table')}.{r.get('to_column')}")

            entry = f"Table: {table_name}"
            if desc:
                entry += f" | Summary: {desc}"
            if pks:
                entry += f" | Primary Keys: [{', '.join(pks)}]"
            if fks:
                entry += f" | Foreign Keys: [{', '.join(fks)}]"
            catalog_lines.append(entry)

        return "\n".join(catalog_lines)

    def get_pruned_schema_string(self, selected_tables: list[str], table_to_columns: dict[str, list[str]] = None) -> str:
        """Formats a high-signal, relation-aware schema string for SQL generation.
        
        Args:
            selected_tables: List of table names to include in the schema.
            table_to_columns: Optional mapping of {table_name: [col1, col2, ...]} for pruned wide tables.
                              If None or a table is missing from map, all columns for that table are included.
        """
        knowledge = self.get_knowledge_structured()
        all_tables = knowledge.get("tables", {})
        if not all_tables:
            raw_struct = self.get_schema_structured()
            all_tables = raw_struct.get("tables", {}) if isinstance(raw_struct, dict) else {}

        relationships = knowledge.get("relationships", [])
        selected_set = {t.strip().lower() for t in selected_tables if t.strip()}
        
        lines = []
        for table_name, info in all_tables.items():
            if table_name.lower() not in selected_set:
                continue

            desc = info.get("description", "").strip()
            pks = set(info.get("primary_keys", []))
            all_cols = info.get("columns", {})
            total_col_count = len(all_cols)

            # Determine which columns to include
            if table_to_columns and table_name in table_to_columns:
                target_cols = [c for c in table_to_columns[table_name] if c in all_cols]
                # Always ensure anchor columns (PKs, FKs, Date/Time columns) are preserved
                fks = [c for c, c_meta in all_cols.items() if c_meta.get("is_fk")]
                date_cols = [c for c in all_cols.keys() if any(k in c.lower() for k in ("date", "time", "year"))]
                anchors = list(dict.fromkeys(list(pks) + fks + date_cols))
                for anchor in anchors:
                    if anchor in all_cols and anchor not in target_cols:
                        target_cols.append(anchor)
                is_pruned = len(target_cols) < total_col_count
            else:
                target_cols = list(all_cols.keys())
                is_pruned = False

            header = f"Table: {table_name}"
            if is_pruned:
                header += f" (Showing {len(target_cols)} of {total_col_count} columns)"
            if desc:
                header += f" (Note: {desc})"
            lines.append(header)

            col_details = []
            for col_name in target_cols:
                col_info = all_cols[col_name]
                c_type = col_info.get("type", "")
                c_desc = col_info.get("description", "").strip()
                
                parts = [col_name]
                flags = []
                if col_name in pks:
                    flags.append("PK")
                if col_info.get("is_fk"):
                    flags.append("FK")

                type_str = f"({c_type})" if c_type else ""
                flag_str = f"[{', '.join(flags)}]" if flags else ""
                
                c_repr = f"{col_name} {type_str}".strip()
                if flag_str:
                    c_repr += f" {flag_str}"
                if c_desc:
                    c_repr += f" [Note: {c_desc}]"
                col_details.append(c_repr)

            lines.append(f"  Columns: [{', '.join(col_details)}]")

        # Append relevant relationships between the selected tables
        relevant_rels = []
        for r in relationships:
            ft = r.get("from_table", "").lower()
            tt = r.get("to_table", "").lower()
            if ft in selected_set and tt in selected_set:
                relevant_rels.append(f"{r.get('from_table')}.{r.get('from_column')} = {r.get('to_table')}.{r.get('to_column')}")

        if relevant_rels:
            lines.append("")
            lines.append("Known Relationships / Join Conditions:")
            for rel in relevant_rels:
                lines.append(f"  - {rel}")

        return "\n".join(lines)

    def get_knowledge_schema(self):
        """Retrieves and formats full schema with descriptions from local JSON (backwards compatible)."""
        knowledge = self.get_knowledge_structured()
        tables = knowledge.get("tables", {})
        if not tables:
            return None
        return self.get_pruned_schema_string(selected_tables=list(tables.keys()))

    # ─────────────────────────────────────────────────────────────
    # Schema Introspection (Live: Tables, PKs, FKs, Types)
    # ─────────────────────────────────────────────────────────────
    def get_schema_structured(self) -> dict:
        """Returns structured dictionary of schema {tables: {tbl: {...}}, relationships: [...]}"""
        if self.mode == "sqlite" and self.sqlite_engine:
            try:
                inspector = inspect(self.sqlite_engine)
                schema_dict = {}
                relationships = []
                for table_name in inspector.get_table_names():
                    pk_info = inspector.get_pk_constraint(table_name)
                    pks = pk_info.get("constrained_columns", []) if pk_info else []
                    
                    fk_info = inspector.get_foreign_keys(table_name) or []
                    fks = []
                    for fk in fk_info:
                        for sc, rc in zip(fk.get("constrained_columns", []), fk.get("referred_columns", [])):
                            fks.append({"constrained_column": sc, "referred_table": fk.get("referred_table"), "referred_column": rc})
                            relationships.append({
                                "from_table": table_name,
                                "from_column": sc,
                                "to_table": fk.get("referred_table"),
                                "to_column": rc
                            })

                    cols = {}
                    for c in inspector.get_columns(table_name):
                        col_name = c['name']
                        cols[col_name] = {
                            "type": str(c.get("type", "TEXT")),
                            "description": "",
                            "is_pk": col_name in pks,
                            "is_fk": any(f["constrained_column"] == col_name for f in fks)
                        }
                    schema_dict[table_name] = {
                        "description": "",
                        "primary_keys": pks,
                        "foreign_keys": fks,
                        "columns": cols
                    }
                return {"tables": schema_dict, "relationships": relationships}
            except Exception as e:
                return f"SCHEMA ERROR: {str(e)}"
        else:
            if not self.database:
                return "SCHEMA ERROR: No database provided in .env"

            engine = self._get_sqlserver_engine()

            # 1. Columns & Data Types
            query_cols = """
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'dbo'
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """
            
            # 2. Primary Keys
            query_pks = """
            SELECT tc.TABLE_NAME, kcu.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu 
                ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME 
                AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY' AND tc.TABLE_SCHEMA = 'dbo'
            """

            # 3. Explicit Foreign Keys
            query_fks = """
            SELECT 
                tp.name AS from_table,
                cp.name AS from_column,
                tr.name AS to_table,
                cr.name AS to_column
            FROM sys.foreign_keys fk
            INNER JOIN sys.tables tp ON fk.parent_object_id = tp.object_id
            INNER JOIN sys.tables tr ON fk.referenced_object_id = tr.object_id
            INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            INNER JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id AND fkc.parent_column_id = cp.column_id
            INNER JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id AND fkc.referenced_column_id = cr.column_id
            WHERE SCHEMA_NAME(tp.schema_id) = 'dbo' AND SCHEMA_NAME(tr.schema_id) = 'dbo'
            """
            try:
                col_df = pd.read_sql(query_cols, engine)
                
                try:
                    pk_df = pd.read_sql(query_pks, engine)
                    pks_by_table = {}
                    for _, r in pk_df.iterrows():
                        pks_by_table.setdefault(r["TABLE_NAME"], []).append(r["COLUMN_NAME"])
                except Exception:
                    pks_by_table = {}

                try:
                    fk_df = pd.read_sql(query_fks, engine)
                    relationships = fk_df.to_dict(orient="records")
                except Exception:
                    relationships = []

                # Build table dictionaries
                schema_dict = {}
                for table_name, group in col_df.groupby("TABLE_NAME"):
                    table_pks = pks_by_table.get(table_name, [])
                    table_fks = [r for r in relationships if r.get("from_table") == table_name]
                    fk_cols = {r.get("from_column") for r in table_fks}

                    cols = {}
                    for _, row in group.iterrows():
                        c_name = row["COLUMN_NAME"]
                        cols[c_name] = {
                            "type": row["DATA_TYPE"],
                            "description": "",
                            "is_pk": c_name in table_pks,
                            "is_fk": c_name in fk_cols
                        }
                    schema_dict[table_name] = {
                        "description": "",
                        "primary_keys": table_pks,
                        "foreign_keys": table_fks,
                        "columns": cols
                    }

                # 4. Implicit Foreign Key Inference (Standard Enterprise heuristic)
                # If explicit FKs are empty, infer relationships where column names match PKs in other tables
                # e.g., transactions.customer_code = customers.customer_code
                all_pk_map = {} # {col_name: table_name}
                for t_name, pks in pks_by_table.items():
                    for pk in pks:
                        all_pk_map[pk.lower()] = (t_name, pk)

                existing_rel_pairs = {(r["from_table"].lower(), r["from_column"].lower(), r["to_table"].lower(), r["to_column"].lower()) for r in relationships}

                for t_name, t_info in schema_dict.items():
                    for c_name, c_info in t_info["columns"].items():
                        c_lower = c_name.lower()
                        # Direct match or singular/plural match (e.g. market_code and markets_code)
                        matched_target = None
                        if c_lower in all_pk_map and all_pk_map[c_lower][0] != t_name:
                            matched_target = all_pk_map[c_lower]
                        elif c_lower.replace("market_code", "markets_code") in all_pk_map and all_pk_map[c_lower.replace("market_code", "markets_code")][0] != t_name:
                            matched_target = all_pk_map[c_lower.replace("market_code", "markets_code")]
                        elif c_lower == "order_date" and "date" in schema_dict and "date" in pks_by_table.get("date", []):
                            matched_target = ("date", "date")

                        if matched_target:
                            target_tbl, target_pk = matched_target
                            rel_key = (t_name.lower(), c_name.lower(), target_tbl.lower(), target_pk.lower())
                            if rel_key not in existing_rel_pairs:
                                existing_rel_pairs.add(rel_key)
                                relationships.append({
                                    "from_table": t_name,
                                    "from_column": c_name,
                                    "to_table": target_tbl,
                                    "to_column": target_pk
                                })
                                c_info["is_fk"] = True

                return {"tables": schema_dict, "relationships": relationships}
            except Exception as e:
                return f"SCHEMA ERROR: {str(e)}"

    def get_schema(self):
        """Deprecated for Semantic Layer, but kept for partial backwards compatibility if needed."""
        nodes = self.get_schema_structured()
        if isinstance(nodes, str): return nodes
        
        schema_str = ""
        for table, info in nodes.items():
            cols = list(info["columns"].keys())
            schema_str += f"Table: {table}, Columns: [{', '.join(cols)}]\n"
        return schema_str

    # 
    # File Upload Security + CSV  In-Memory SQLite Loader
    # 
    def sanitize_filename(self, filename: str) -> str:
        """Strip special characters and enforce max length."""
        name = re.sub(r"[^\w\-.]", "_", filename)  # keep alphanumeric, dash, dot
        name = name[:MAX_FILENAME_LENGTH]            # enforce max length
        return name

    def validate_upload_extension(self, filename: str) -> tuple[bool, str]:
        """Returns (is_valid, error_message)."""
        _, ext = os.path.splitext(filename.lower())
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            return False, f"File type '{ext}' not allowed. Upload CSV, XLSX, or XLS only."
        return True, ""

    def validate_upload_file(self, file_path: str) -> tuple[bool, str]:
        """Validates file size and extension. Returns (is_valid, message)."""
        if not os.path.exists(file_path):
            return False, "File not found"
        size = os.path.getsize(file_path)
        size_mb = size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            return False, f"File too large: {size_mb:.1f} MB. Max allowed is {MAX_UPLOAD_SIZE_MB} MB."
        filename = os.path.basename(file_path)
        ok, msg = self.validate_upload_extension(filename)
        if not ok:
            return False, msg
        return True, ""

    def load_csv_to_memory(self, file_path):
        """Loads a CSV into SQLite to allow querying it instead of SQL Server.
        Auto-detects delimiter (comma, semicolon, tab, pipe, etc.).
        """
        import csv

        # Validate upload first (size + extension)
        ok, msg = self.validate_upload_file(file_path)
        if not ok:
            logger.warning(f"[Upload] Validation failed: {msg} | file: {file_path}")
            # Import exception lazily to avoid circular package imports
            from src.agent.exceptions import SecurityError
            raise SecurityError("Uploaded file rejected.", internal_detail=msg)

        self.mode = "sqlite"
        if not self.sqlite_engine:
            self.sqlite_engine = create_engine(self.in_memory_db, connect_args={"check_same_thread": False})

        # Auto-detect delimiter
        with open(file_path, "r", encoding="utf-8-sig") as f:
            sample = f.read(8192)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
            except csv.Error:
                sep = ","

        logger.info(f"[Upload] Detected delimiter: '{sep}' for file: {file_path}")
        df = pd.read_csv(file_path, sep=sep)

        raw_name = os.path.basename(file_path)
        table_name = self.sanitize_filename(raw_name).split(".")[0].replace(" ", "_").lower()
        df.columns = df.columns.str.replace(" ", "_").str.lower()

        # Auto-detect & convert date columns
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    non_null_count = df[col].notna().sum()
                    if non_null_count > 0:
                        success_rate = converted.notna().sum() / non_null_count
                        if success_rate >= 0.8:
                            df[col] = converted
                            logger.info(f"[Upload] Auto-detected date column: '{col}' ({success_rate:.0%} parsed)")
                except Exception:
                    pass

        df.to_sql(table_name, con=self.sqlite_engine, index=False, if_exists="replace")
        logger.info(f"[Upload] Loaded {len(df)} rows  {len(df.columns)} cols into table '{table_name}'")
        print(f"[Upload] Loaded {len(df)} rows x {len(df.columns)} columns into table: '{table_name}'")
        print(f"   Columns: {list(df.dtypes.to_dict().items())}")
        
        # Auto-sync knowledge base for the new ad-hoc table
        self.sync_knowledge_base()
        
        return table_name
