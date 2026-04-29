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
    def sync_knowledge_base(self):
        """Fetches latest schema and merges with existing descriptions in local JSON."""
        logger.info(f"[Knowledge] Syncing knowledge base for mode: {self.mode}...")
        
        # 1. Fetch live structured schema
        new_tables = self.get_schema_structured()
        if isinstance(new_tables, str) and "ERROR" in new_tables:
            return False, new_tables

        # 2. Load existing knowledge (to preserve descriptions)
        existing_knowledge = {}
        if os.path.exists(KNOWLEDGE_PATH):
            try:
                with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                    existing_knowledge = json.load(f).get("tables", {})
            except Exception:
                logger.warning("[Knowledge] Could not load existing knowledge for merging.")

        # 3. Merge: Preserve existing descriptions for tables and columns
        for table_name, table_info in new_tables.items():
            if table_name in existing_knowledge:
                # Table exists? Keep table description
                table_info["description"] = existing_knowledge[table_name].get("description", "")
                
                # Column check: Keep column descriptions
                existing_cols = existing_knowledge[table_name].get("columns", {})
                for col_name, col_info in table_info["columns"].items():
                    if col_name in existing_cols:
                        col_info["description"] = existing_cols[col_name].get("description", "")

        knowledge = {
            "mode": self.mode,
            "database": self.database if self.mode == "sqlserver" else "sqlite_memory",
            "tables": new_tables,
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

    def get_knowledge_schema(self):
        """Retrieves and formats schema with descriptions from local JSON."""
        if not os.path.exists(KNOWLEDGE_PATH):
            return None

        try:
            with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                knowledge = json.load(f)
            
            if knowledge.get("mode") == self.mode:
                if self.mode == "sqlserver" and knowledge.get("database") != self.database:
                    return None
                
                # Format the structured JSON back into an enriched string for the LLM
                tables = knowledge.get("tables", {})
                formatted_lines = []
                for table_name, info in tables.items():
                    desc = info.get("description", "").strip()
                    table_line = f"Table: {table_name}"
                    if desc:
                        table_line += f" (Note: {desc})"
                    
                    col_details = []
                    for col_name, col_info in info.get("columns", {}).items():
                        c_desc = col_info.get("description", "").strip()
                        c_type = col_info.get("type", "")
                        c_str = col_name
                        if c_type:
                            c_str += f" ({c_type})"
                        if c_desc:
                            c_str += f" [Note: {c_desc}]"
                        col_details.append(c_str)
                    
                    table_line += f", Columns: [{', '.join(col_details)}]"
                    formatted_lines.append(table_line)
                
                logger.info("[Knowledge] Enriched schema retrieved from local Knowledge Base.")
                return "\n".join(formatted_lines)
        except Exception as e:
            logger.warning(f"[Knowledge] Failed to parse knowledge base: {e}")
            
        return None

    # 
    # Schema Introspection (Live)
    # 
    def get_schema_structured(self) -> dict:
        """New: Returns a structured dictionary of the schema {table: {description, columns: {col: {type, description}}}}"""
        if self.mode == "sqlite" and self.sqlite_engine:
            try:
                inspector = inspect(self.sqlite_engine)
                schema_dict = {}
                for table_name in inspector.get_table_names():
                    cols = {}
                    for c in inspector.get_columns(table_name):
                        cols[c['name']] = {
                            "type": str(c.get("type", "TEXT")),
                            "description": ""
                        }
                    schema_dict[table_name] = {
                        "description": "",
                        "columns": cols
                    }
                return schema_dict
            except Exception as e:
                return f"SCHEMA ERROR: {str(e)}"
        else:
            if not self.database:
                return "SCHEMA ERROR: No database provided in .env"

            query = """
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'dbo'
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """
            try:
                engine = self._get_sqlserver_engine()
                df = pd.read_sql(query, engine)
                schema_dict = {}
                for table_name, group in df.groupby("TABLE_NAME"):
                    cols = {}
                    for _, row in group.iterrows():
                        cols[row["COLUMN_NAME"]] = {
                            "type": row["DATA_TYPE"],
                            "description": ""
                        }
                    schema_dict[table_name] = {
                        "description": "",
                        "columns": cols
                    }
                return schema_dict
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
