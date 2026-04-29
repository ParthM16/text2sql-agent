"""
Agent Workflow  Orchestrator
==============================
Manages the agentic execution graph: intent detection  table selection 
attribute matching  SQL generation  execution  error correction loop 
natural language answer. Also handles conversation memory for follow-ups
and chart rendering.
"""
import os
import pandas as pd
import threading
from time import perf_counter

from src.agent.nodes import AgentNodes
from src.db.helper import DBHelper
from src.retrieval.vector_store import VectorStore
from src.utils.metrics import PersistentMetrics, SessionMetrics
from src.utils.logger import get_logger


class TextToSQLAgent:
    """Production-grade Text-to-SQL agent with multi-node agentic workflow."""

    def __init__(self, db_helper: DBHelper, vector_store: VectorStore):
        self.db = db_helper
        self.vs = vector_store
        self.nodes = AgentNodes()

        #  Conversation Memory 
        # Stores the last successful query context so follow-up
        # questions ("show chart", "break it down by region") can
        # reuse the cached data without hitting the DB again.
        self.last_query = None
        self.last_sql = None
        self.last_results = None   # list[dict] from DB
        self.last_df = None        # pandas DataFrame for charting
        self.last_mode = db_helper.mode  # Track if user switched CSV <-> SQL Server

        #  UUI Session Cache 
        # Stores discovered units per-table so Shadow Discovery
        # only hits the DB once per session. Flushed on mode switch.
        self._unit_cache = {}  # {table_name: set(units)}

        # Metrics + logger
        self.metrics = SessionMetrics()
        self.logger = get_logger(__name__)

        #  Seed Few-Shot Memory 
        # Addresses Gap Analysis P0: Vector DB starts empty unless seeded
        try:
            if self.vs.few_shot_index.ntotal == 0:
                # Up 3 directories: src/agent/workflow.py -> /data/few_shot_examples.json
                seed_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "few_shot_examples.json")
                if os.path.exists(seed_path):
                    import json
                    with open(seed_path, "r", encoding="utf-8") as f:
                        seeds = json.load(f)
                        self.vs.add_few_shots(seeds)
                        self.logger.info(f"Seeded Vector Store with {len(seeds)} contextual examples.")
        except Exception as e:
            self.logger.warning(f"Could not load few-shot seeds: {e}")

    def run(self, user_query: str, ui_callback=None, app_language="en", stop_event=None) -> tuple[str, list, dict, pd.DataFrame]:
        """Main entry point. Returns (answer_text, log_lines, chart_config, data_frame)."""
        overall_start = perf_counter()
        log = []
        
        def notify_ui(msg):
            if ui_callback: ui_callback(msg)

        def print_terminal(node_name):
            import time
            print(f"[\033[94m{time.strftime('%H:%M:%S')}\033[0m] ⏳ Running Node: {node_name}", flush=True)

            
        notify_ui("Understanding Request")
        print_terminal("Input Guardrail")

        # Pre-initialize ALL keys to 0.0 to ensure a consistent JSON structure in the dashboard
        node_timings = {
            "01_intent": 0.0,
            "02_knowledge": 0.0,
            "03_table_selection": 0.0,
            "04_few_shot": 0.0,
            "05_attr_match": 0.0,
            "06_sql_generation": 0.0,
            "07_sql_exec": 0.0,
            "08_sql_correct": 0.0,
            "09_chart_config": 0.0,
            "10_nlg": 0.0
        }
        tokens_agg = {"prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0}

        #  Guard: API credentials 
        has_vertex = bool(os.getenv("PROJECT_ID", ""))
        has_api_key = os.getenv("GOOGLE_API_KEY", "") not in ("", "your_api_key_here")
        if not has_vertex and not has_api_key:
            return "Error: No API credentials configured. Set PROJECT_ID (Vertex AI) or GOOGLE_API_KEY in .env.", log, None, None

        #  Guard: Input validation 
        trimmed = user_query.strip()
        if not trimmed:
            return "Please enter a question.", log, None, None
        if len(trimmed) > 500:
            return "Query too long (max 500 characters). Please shorten your question.", log, None, None

        # Guard: Minimum Query Quality Check
        alpha_chars = sum(1 for c in trimmed if c.isalpha())
        if alpha_chars < 2:
            duration = perf_counter() - overall_start
            log.append(f"[Node: Input Guardrail] Blocked: Query too short or meaningless. ({duration:.3f}s)")
            node_timings["00_input_guardrail"] = duration
            return "Please provide a more descriptive question (e.g., 'show me total sales by year').", log, None, None

        #  Guard: Pre-flight safety check (Input Guardrail) 
        import re
        import difflib
        
        # --- Targeted Pre-AI Spell Correction ---
        # Only correct words close to dangerous keywords so we don't break domain names (like 'walmart' -> 'walnut')
        critical_keywords = [
            "DROP", "TABLE", "TABLES", "DATABASE", "DATABASES", "VIEW", "VIEWS",
            "DELETE", "FROM", "RECORD", "RECORDS", "INSERT", "INTO",
            "UPDATE", "SET", "CREATE",
            "ALTER", "TRUNCATE", "IGNORE", "PREVIOUS", "SYSTEM", "PROMPT", "BYPASS", "INSTRUCTIONS",
            "EXEC", "EXECUTE", "GRANT", "REVOKE"
        ]
        
        corrected_words = []
        for w in trimmed.split():
            clean_original = ''.join(c for c in w if c.isalnum())
            clean_w = clean_original.upper()
            if len(clean_w) > 2 and clean_w not in critical_keywords:
                matches = difflib.get_close_matches(clean_w, critical_keywords, n=1, cutoff=0.85)
                if matches:
                    matched_word = matches[0].lower() if w.islower() else matches[0]
                    if clean_original:
                        w = w.replace(clean_original, matched_word)
            corrected_words.append(w)
            
        trimmed = " ".join(corrected_words)
        trimmed_upper = trimmed.upper()
        # --- End Spell Correction ---
        
        # 1. Machine Understanding Language / Encoded Injection Check
        is_encoded = False
        
        # Binary (Catches ANY sequence of 8+ binary digits)
        if re.search(r"[01]{4,}", trimmed) or re.fullmatch(r"^[01\s]+$", trimmed):
            is_encoded = True
            
        # Hexadecimal (Spaced or Continuous)
        hex_candidates = re.findall(r"\b(?:[A-Fa-f0-9]{2}\s*){6,}\b", trimmed)
        for candidate in hex_candidates:
            if re.search(r"[A-Fa-f]", candidate, re.IGNORECASE):
                is_encoded = True
                break
                
        # Base64 (Catches continuous strings of 20+ chars ending in =)
        if re.search(r"(?:\b|\s|^)[A-Za-z0-9+/]{20,}={1,2}(?:\s|$)", trimmed):
            is_encoded = True
            
        # NCR Decimal / Hexadecimal (HTML Entities like &#108; or &#x6C;)
        if re.search(r"(?:&#[xX]?[A-Fa-f0-9]+;\s*){3,}", trimmed):
            is_encoded = True

        if is_encoded:
            duration = perf_counter() - overall_start
            log.append(f"[Node: Input Guardrail] Blocked: Machine/Encoded sequence detected. ({duration:.3f}s)")
            node_timings["00_input_guardrail"] = duration
            return "Security Policy Violation: Machine understanding languages (Binary, Hexadecimal, Base64) are not allowed. Please use human-readable natural language.", log, None, None

        # 1.5 Natural Language Detection & Decimal Bypass Catch
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            detected_lang = detect(trimmed)
            
            # For queries longer than 2 words, enforce strict language matching
            if len(trimmed.split()) > 2 and detected_lang != app_language:
                duration = perf_counter() - overall_start
                print_terminal(f"Language Mismatch Block (Got: {detected_lang}, Expected: {app_language})")
                log.append(f"[Node: Input Guardrail] Blocked: Language mismatch. ({duration:.3f}s)")
                node_timings["00_input_guardrail"] = duration
                return "Security Policy Violation: Your input does not match the selected support language.", log, None, None
                
        except Exception as e:
            # Langdetect fails on non-linguistic inputs (URLs, decimal sequences, etc.)
            duration = perf_counter() - overall_start
            print_terminal(f"Language Detection Failure: {type(e).__name__}: {e}")
            log.append(f"[Node: Input Guardrail] Blocked: Unrecognized input. ({duration:.3f}s)")
            node_timings["00_input_guardrail"] = duration
            return "Security Policy Violation: Your input does not match the selected support language.", log, None, None

        # 2. Forbidden Operations & Prompt Injection Check
        dangerous_patterns = {
            r"\bDROP\b\s+(?:\w+\s+)*\b(?:TABLES?|DATABASES?|VIEWS?)\b": "DROP OPERATION",
            r"\bDELETE\b\s+(?:\w+\s+)*\b(?:FROM|RECORDS?)\b": "DELETE OPERATION",
            r"\bINSERT\b\s+(?:\w+\s+)*\bINTO\b": "INSERT OPERATION",
            r"\bUPDATE\b\s+(?:\w+\s+)*\bSET\b": "UPDATE OPERATION",
            r"\bCREATE\b\s+(?:\w+\s+)*\b(?:TABLES?|DATABASES?|VIEWS?)\b": "CREATE OPERATION",
            r"\bALTER\b\s+(?:\w+\s+)*\bTABLES?\b": "ALTER OPERATION",
            r"\bTRUNCATE\b\s+(?:\w+\s+)*\bTABLES?\b": "TRUNCATE OPERATION",
            r"IGNORE ALL PREVIOUS": "PROMPT INJECTION", 
            r"FORGET ALL": "PROMPT INJECTION", 
            r"SYSTEM PROMPT": "PROMPT INJECTION", 
            r"BYPASS INSTRUCTIONS": "PROMPT INJECTION"
        }
        
        blocked_reason = None
        for pattern, human_name in dangerous_patterns.items():
            if re.search(pattern, trimmed_upper):
                blocked_reason = human_name
                break
                
        # 3. Strict Single-Word DML/DDL Check (avoiding common English words like DROP, CREATE, UPDATE)
        if not blocked_reason:
            tokens = set(re.findall(r"\b[A-Z]+\b", trimmed_upper))
            strict_words = {"DELETE", "INSERT", "TRUNCATE", "EXEC", "EXECUTE", "GRANT", "REVOKE", "ALTER"}
            found = tokens & strict_words
            if found:
                blocked_reason = next(iter(found))

        if blocked_reason:
            duration = perf_counter() - overall_start
            log.append(f"[Node: Input Guardrail] Blocked: {blocked_reason} ({duration:.3f}s)")
            node_timings["00_input_guardrail"] = duration
            return "Security Policy Violation: Your query contains forbidden keywords or attempts to bypass system instructions.", log, None, None

        #  Detect Mode Switch (Flush memory if shifted) 
        current_mode = self.db.mode
        if current_mode != self.last_mode:
            self.logger.info(f" Mode shift detected ({self.last_mode} -> {current_mode}). Flushing memory.")
            self.last_query = None
            self.last_sql = None
            self.last_results = None
            self.last_df = None
            self._unit_cache = {}  # Flush unit cache on mode switch
            self.last_mode = current_mode

        #  Intent Detection (enables follow-ups & charts) 
        intent = "NEW_QUERY"
        if self.last_query is not None:
            print_terminal("Intent Detection")
            t0 = perf_counter()
            try:
                intent_res, usage = self.nodes.intent_detection(
                user_query,
                last_query=self.last_query,
                last_sql=self.last_sql,
                last_results=self.last_results,
                last_df=self.last_df,
                )
            except Exception as e:
                # Node failure  record and return user-friendly message
                from src.agent.exceptions import AgentError
                if isinstance(e, AgentError):
                    user_msg = e.user_message
                    self.logger.warning(f"Intent node error: {e.internal_detail}")
                else:
                    user_msg = "Internal error during intent detection. Try again later."
                    self.logger.exception("Intent detection failed")
                # persist failure
                try:
                    PersistentMetrics.append_query({
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "user_query": user_query,
                        "generated_sql": None,
                        "intent": "INTENT_ERROR",
                        "success": False,
                        "tokens_used": None,
                        "response_time_ms": None,
                        "node_timings": node_timings,
                    })
                except Exception:
                    self.logger.exception("Failed to persist intent error")
                self.last_query = user_query
                return user_msg, log, None, None
            duration = perf_counter() - t0
            node_timings["01_intent"] = duration
            log.append(f"[Node: Intent Detection] Detected intent: {intent_res} ({duration:.3f}s)")
            intent = intent_res
            if usage:
                pt = usage.get("prompt_tokens") or 0
                rt = usage.get("response_tokens") or 0
                tokens_agg["prompt_tokens"] += pt
                tokens_agg["response_tokens"] += rt
                tokens_agg["total_tokens"] += usage.get("total_tokens") or (pt + rt)
                self.metrics.add_tokens("intent", pt, rt)
        else:
            # No previous context  use fast keyword pre-check for EXPLORATORY
            explore_kw = ['what can i', 'what analysis', 'help me', 'what data', 'suggest', 'explore', 'what insight', 'what kind']
            if any(kw in user_query.lower() for kw in explore_kw):
                intent = "EXPLORATORY"
                log.append("[Node: Intent Detection] No context + exploratory keywords detected. Treating as EXPLORATORY.")
            else:
                log.append("[Node: Intent Detection] No previous context. Treating as NEW_QUERY.")
            node_timings["01_intent"] = 0.0

        #  EXPLORATORY path: schema-only suggestions 
        if intent == "EXPLORATORY":
            notify_ui("Generating Suggestions")
            print_terminal("Schema Retrieval")
            t0 = perf_counter()
            t_know = perf_counter()
            full_schema = self.db.get_knowledge_schema()
            if full_schema is None:
                ok, err = self.db.sync_knowledge_base()
                if not ok:
                    return f"Database knowledge sync failed: {err}", log, None, None
                full_schema = self.db.get_knowledge_schema()
            node_timings["02_knowledge"] = perf_counter() - t_know
            log.append("[Node: Schema Retrieval] Retrieved from persistent Knowledge Base (Cache).")
            
            try:
                print_terminal("Exploratory Answer Generator")
                answer, usage = self.nodes.exploratory_answer(user_query, full_schema)
            except Exception as e:
                from src.agent.exceptions import AgentError
                if isinstance(e, AgentError):
                    answer = e.user_message
                else:
                    answer = "I couldn't generate suggestions right now. Please try again."
                    self.logger.exception("Exploratory answer failed")
            duration = perf_counter() - t0
            node_timings["10_nlg"] = duration
            log.append(f"[Node: Exploratory Answer] Schema-only suggestions generated. ({duration:.3f}s)")
            if usage:
                pt = usage.get("prompt_tokens") or 0
                rt = usage.get("response_tokens") or 0
                tokens_agg["prompt_tokens"] += pt
                tokens_agg["response_tokens"] += rt
                tokens_agg["total_tokens"] += usage.get("total_tokens") or (pt + rt)
            try:
                PersistentMetrics.append_query({
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "user_query": user_query,
                    "generated_sql": None,
                    "intent": "EXPLORATORY",
                    "success": True,
                    "tokens_used": tokens_agg,
                    "response_time_sec": round(perf_counter() - overall_start, 3),
                    "node_timings": node_timings,
                })
                self.metrics.add_query_result(True)
            except Exception:
                self.logger.exception("Failed to persist exploratory query")
            return answer, log, None, None

        #  CHART_REQUEST path: use cached data 
        chart_config = None
        if intent == "CHART_REQUEST":
            notify_ui("Building Chart")
            print_terminal("Chart Builder")
            if self.last_df is None:
                msg = ("I'm sorry, I recently reset my session (likely due to a data source change) "
                       "and I'm not sure which data you'd like me to chart. "
                       "Could you please describe what you'd like to see in a chart?")
                log.append("[Node: Chart Builder] Context missing. Returning clarification request.")
                return msg, log, None, None
            log.append("[Node: Chart Builder] Using cached data from previous query.")
            t0 = perf_counter()
            try:
                chart_config, usage = self.nodes.chart_config(user_query, self.last_df)
            except Exception as e:
                from src.agent.exceptions import AgentError
                if isinstance(e, AgentError):
                    user_msg = e.user_message
                    self.logger.warning(f"Chart node error: {e.internal_detail}")
                else:
                    user_msg = "Internal error building chart. Try again later."
                    self.logger.exception("Chart config failed")
                try:
                    PersistentMetrics.append_query({
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "user_query": user_query,
                        "generated_sql": None,
                        "intent": "CHART_ERROR",
                        "success": False,
                        "tokens_used": None,
                        "response_time_ms": None,
                        "node_timings": node_timings,
                    })
                except Exception:
                    self.logger.exception("Failed to persist chart error")
                self.last_query = user_query
                return user_msg, log, None, None
            duration = perf_counter() - t0
            node_timings["09_chart_config"] = duration
            log.append(f"[Node: Chart Builder] Config: {chart_config} ({duration:.3f}s)")
            if usage:
                pt = usage.get("prompt_tokens") or 0
                rt = usage.get("response_tokens") or 0
                tokens_agg["prompt_tokens"] += pt
                tokens_agg["response_tokens"] += rt
                tokens_agg["total_tokens"] += usage.get("total_tokens") or (pt + rt)
                self.metrics.add_tokens("chart", pt, rt)
            answer = f"Here is the **{chart_config['chart_type']} chart** for: *{chart_config['title']}*"
            total_time_sec = round(perf_counter() - overall_start, 3)
            try:
                PersistentMetrics.append_query({
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "user_query": user_query,
                    "generated_sql": None,
                    "intent": intent,
                    "success": True,
                    "tokens_used": tokens_agg,
                    "response_time_sec": total_time_sec,
                    "node_timings": node_timings,
                })
                self.metrics.add_query_result(True)
                self.logger.info("[Analytics] Persisted chart-only request to history.")
            except Exception:
                self.logger.exception("Failed to persist chart history")
            
            # Return immediately for chart-only refreshes
            return answer, log, chart_config, self.last_df

        #  FOLLOW_UP path: re-query with context 
        actual_query = user_query
        if intent == "FOLLOW_UP":
            if not self.last_query:
                msg = ("I'm sorry, I've lost track of our previous conversion context. "
                       "Could you please re-state your question with full details?")
                log.append("[Node: Follow-Up] Context missing. Returning clarification request.")
                return msg, log, None, None
            actual_query = f"{user_query} (Context: the previous question was '{self.last_query}')"
            log.append("[Node: Follow-Up] Enriched query with previous context.")

        #  Standard NEW_QUERY / FOLLOW_UP pipeline 
        notify_ui("Searching Database")
        #  Persistent Knowledge Engine: Check cache before querying DB
        t_know = perf_counter()
        full_schema = self.db.get_knowledge_schema()
        
        if full_schema is None:
            self.logger.info("[Knowledge] Cache missing or invalid. Triggering auto-sync...")
            ok, err = self.db.sync_knowledge_base()
            if not ok:
                log.append(f"[Node: Knowledge Sync] FAILED: {err}")
                return f"Database knowledge sync failed. Details: {err}", log, None, None
            full_schema = self.db.get_knowledge_schema()
            log.append("[Node: Schema Retrieval] Auto-synced and retrieved from Knowledge Base.")
        else:
            log.append(f"[Node: Schema Retrieval] Retrieved from persistent Knowledge Base (Cache).")
        
        node_timings["02_knowledge"] = perf_counter() - t_know

        # Node 1: Table Selection
        print_terminal("Table Selection")
        t0 = perf_counter()
        try:
            relevant_tables, usage = self.nodes.table_selection(full_schema, actual_query)
        except Exception as e:
            from src.agent.exceptions import AgentError
            if isinstance(e, AgentError):
                user_msg = e.user_message
                self.logger.warning(f"Table selection error: {e.internal_detail}")
            else:
                user_msg = "Internal error selecting tables. Try again later."
                self.logger.exception("Table selection failed")
            try:
                PersistentMetrics.append_query({
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "user_query": user_query,
                    "generated_sql": None,
                    "intent": "TABLE_SELECT_ERROR",
                    "success": False,
                    "tokens_used": None,
                    "response_time_ms": None,
                    "node_timings": node_timings,
                })
            except Exception:
                self.logger.exception("Failed to persist table selection error")
            self.last_query = user_query
            return user_msg, log, None, None
        duration = perf_counter() - t0
        node_timings["03_table_selection"] = duration
        log.append(f"[Node: Table Selection] {relevant_tables} ({duration:.3f}s)")

        #  Schema Anchor Logic (Semantic Validation) 
        # We only block a query if the Table Selection node says "NONE" (meaning it found no relevant tables)
        # AND it's a new query (not a follow-up that might refer to existing context/charts).
        is_none = "NONE" in relevant_tables.upper() or not relevant_tables.strip()
        if is_none and intent == "NEW_QUERY":
            msg = ("I'm sorry, I couldn't find any relevant data for that request in the current database. "
                   "Could you please rephrase or specify which part of the retail data you are interested in?")
            log.append("[Node: Schema Anchor] Unanswerable query (No relevant tables found). Blocking.")
            return msg, log, None, None

        if usage:
            pt = usage.get("prompt_tokens") or 0
            rt = usage.get("response_tokens") or 0
            tokens_agg["prompt_tokens"] += pt
            tokens_agg["response_tokens"] += rt
            tokens_agg["total_tokens"] += usage.get("total_tokens") or (pt + rt)
            self.metrics.add_tokens("table_select", pt, rt)

        # Build reduced schema string
        reduced_schema = ""
        for t in relevant_tables.split(","):
            t = t.strip()
            for line in full_schema.split("\n"):
                # Match "Table: transactions" regardless of what follows (comma, space, paren)
                if f"Table: {t}" in line or f"Table: {t.lower()}" in line.lower():
                    reduced_schema += line + "\n"
        if not reduced_schema.strip():
            reduced_schema = full_schema  # fallback

        #  Pre-Generation Shadow Discovery & UUI 
        print_terminal("Shadow Discovery (UUI)")
        t_unit = perf_counter()
        unit_context = {"collision_detected": False}
        unit_conversion_rules = None
        
        # Relevance check: only scan if query contains unit keywords
        unit_keywords = ['sales', 'amount', 'profit', 'margin', 'price', 'cost', 'weight', 'qty', 'total', 'average', 'sum', 'conversion', 'currency', 'inr', 'yen', 'usd']
        is_unit_relevant = any(kw in actual_query.lower() for kw in unit_keywords)
        
        if is_unit_relevant:
            discovered_units = {}
            target_tables = [t.strip() for t in relevant_tables.split(",") if t.strip()]
            
            # Check cache
            cache_hit = False
            for tbl in target_tables:
                if tbl.lower() in self._unit_cache:
                    discovered_units = self._unit_cache[tbl.lower()]
                    cache_hit = True
                    log.append(f"[Node: Shadow Discovery] Cache hit for '{tbl}': {discovered_units}")
                    break
            
            if not cache_hit:
                unit_cols = ['currency', 'unit', 'unit_name', 'symbol', 'uom']
                for tbl in target_tables:
                    for col in unit_cols:
                        if col in str(reduced_schema).lower():
                            check_sql = f"SELECT {col}, COUNT(*) as cnt FROM {tbl} WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY cnt DESC"
                            try:
                                check_results, err = self.db.execute_query(check_sql)
                                if check_results:
                                    for r in check_results:
                                        vals = list(r.values())
                                        if len(vals) >= 2 and vals[0]:
                                            discovered_units[str(vals[0]).upper()] = int(vals[1])
                            except Exception:
                                pass
                    if discovered_units:
                        self._unit_cache[tbl.lower()] = discovered_units.copy()
                        log.append(f"[Node: Shadow Discovery] DB scan for '{tbl}': {discovered_units} (cached for session)")
                        break
                        
            if len(discovered_units) > 1:
                import re
                unit_context["collision_detected"] = True
                unit_context["units_found"] = list(discovered_units.keys())
                
                # Deterministic Target Unit Selection (NO LLM REQUIRED)
                target_unit = None
                
                # Rule 1: User Preference
                for unit in discovered_units.keys():
                    if re.search(rf'\b{unit.lower()}\b', actual_query.lower()):
                        target_unit = unit
                        break
                
                # Rule 2: Majority
                if not target_unit:
                    sorted_units = sorted(discovered_units.items(), key=lambda x: x[1], reverse=True) # Descending (highest first)
                    target_unit = sorted_units[0][0]
                
                # Find minority to convert
                minority_units = [u for u in discovered_units.keys() if u != target_unit]
                minority = minority_units[0] if minority_units else target_unit
                
                # Fetch Web Factor
                from src.utils.web_service import WebConversionService
                search_q = f"convert {minority} to {target_unit}"
                web_factor = WebConversionService.get_conversion_factor(search_q)
                
                unit_context["live_web_factor"] = web_factor
                unit_context["caution_message"] = f"Multiple currency units ({', '.join(unit_context['units_found'])}) were detected. The data has been mathematically normalized to {target_unit} before aggregation."
                
                log.append(f"[Node: Pre-Gen UUI] Collision detected ({unit_context['units_found']}). Auto-selected Target: {target_unit}. Web Factor: {web_factor}")
                
                # Build the conversion rules for the SQL Generation prompt
                unit_conversion_rules = f"""
        MANDATORY UNIT NORMALIZATION RULES:
        We detected mixed currencies/units in this table ({', '.join(unit_context['units_found'])}).
        You MUST perform RECORD-LEVEL mathematical normalization inside your SUM/AVG aggregations.
        Target Unit for all calculations: '{target_unit}'
        Conversion Rule: {web_factor}
        
        Example format: SUM(CASE WHEN currency = '{minority}' THEN (sales_amount * 0.58) WHEN currency = '{target_unit}' THEN sales_amount ELSE sales_amount END)
        Do NOT use MAX(currency) or group by currency. Only output the normalized aggregated numbers.
        """
        
        node_timings["09a_unit_validation_total"] = perf_counter() - t_unit

        # Node 2: Few Shot Retrieval
        print_terminal("Few-Shot Retrieval")
        t_ret = perf_counter()
        few_shots = self.vs.retrieve_few_shots(actual_query, k=2)
        node_timings["04_few_shot"] = perf_counter() - t_ret
        log.append(f"[Node: Few Shot] Retrieved {len(few_shots)} examples.")

        # Node 3: Attribute Matching (semantic typo correction)
        print_terminal("Attribute Matching")
        t_attr = perf_counter()
        matches = []
        closest_match = self.vs.match_attribute(actual_query)
        if closest_match:
            matches.append(closest_match)
            log.append(f"[Node: Attributes] Corrected: {closest_match['col']}  {closest_match['val']}")
        else:
            log.append("[Node: Attributes] No semantic typos found.")
        node_timings["05_attr_match"] = perf_counter() - t_attr

        # Node 4: SQL Generation
        print_terminal("SQL Generation")
        notify_ui("Writing SQL Query")
        t0 = perf_counter()
        try:
            sql, usage = self.nodes.sql_generation(
                actual_query, 
                reduced_schema, 
                few_shots, 
                matches, 
                db_mode=self.db.mode,
                unit_conversion_rules=unit_conversion_rules
            )
        except Exception as e:
            from src.agent.exceptions import AgentError
            if isinstance(e, AgentError):
                user_msg = e.user_message
                self.logger.warning(f"SQL generation error: {e.internal_detail}")
            else:
                user_msg = "Internal error generating SQL. Try again later."
                self.logger.exception("SQL generation failed")
            try:
                PersistentMetrics.append_query({
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "user_query": user_query,
                    "generated_sql": None,
                    "intent": "SQL_GEN_ERROR",
                    "success": False,
                    "tokens_used": None,
                    "response_time_ms": None,
                    "node_timings": node_timings,
                })
            except Exception:
                self.logger.exception("Failed to persist sql generation error")
            self.last_query = user_query
            return user_msg, log, None, None
        duration = perf_counter() - t0
        node_timings["06_sql_generation"] = duration

        # Intercept NOT_FOUND schema hallucinaton protection
        if sql.startswith("NOT_FOUND:"):
            msg = sql.replace("NOT_FOUND:", "").strip()
            log.append(f"[Node: SQL Generation] Schema missing elements. Agent aborted and asked user for clarification.")
            self.logger.info("Schema missing elements short-circuit.")
            
            # Save context so the user can answer clarification questions
            # We preserve last_results/last_df from the previous success to maintain session memory
            self.last_query = user_query
            
            try:
                PersistentMetrics.append_query({
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "user_query": user_query,
                    "generated_sql": None,
                    "intent": "MISSING_SCHEMA",
                    "success": False,
                    "tokens_used": None,
                    "response_time_ms": None,
                    "node_timings": node_timings,
                })
            except Exception:
                pass
            return msg, log, None, None

        log.append(f"[Node: SQL Generation]\n{sql} ({duration:.3f}s)")
        if usage:
            pt = usage.get("prompt_tokens") or 0
            rt = usage.get("response_tokens") or 0
            tokens_agg["prompt_tokens"] += pt
            tokens_agg["response_tokens"] += rt
            tokens_agg["total_tokens"] += usage.get("total_tokens") or (pt + rt)
            self.metrics.add_tokens("sql_gen", pt, rt)

        # Node 5 + 6: Execution + Auto-Correction Loop
        max_retries = 3
        success = False
        db_results = None

        for attempt in range(max_retries):
            print_terminal(f"SQL Execution (Attempt {attempt + 1})")
            exec_t0 = perf_counter()
            results, error = self.db.execute_query(sql)
            exec_duration = perf_counter() - exec_t0
            node_timings.setdefault("07_sql_exec", 0)
            node_timings["07_sql_exec"] += exec_duration
            self.metrics.add_timing("sql_exec", exec_duration)
            if error:
                log.append(f"[Node: SQL Exec Failed] Attempt {attempt + 1}. Error: {error}")
                if attempt < max_retries - 1:
                    log.append("[Node: SQL Correction] Auto-fixing...")
                    print_terminal("SQL Correction")
                    t0 = perf_counter()
                    try:
                        sql, usage = self.nodes.sql_correction(actual_query, sql, error, reduced_schema, db_mode=self.db.mode)
                    except Exception as e:
                        from src.agent.exceptions import AgentError
                        if isinstance(e, AgentError):
                            user_msg = e.user_message
                            self.logger.warning(f"SQL correction error: {e.internal_detail}")
                        else:
                            user_msg = "Internal error during SQL correction. Try again later."
                            self.logger.exception("SQL correction failed")
                        try:
                            PersistentMetrics.append_query({
                                "timestamp": pd.Timestamp.now().isoformat(),
                                "user_query": user_query,
                                "generated_sql": sql,
                                "intent": "SQL_CORRECT_ERROR",
                                "success": False,
                                "tokens_used": tokens_agg,
                                "response_time_sec": round(perf_counter() - overall_start, 3),
                                "node_timings": node_timings,
                            })
                        except Exception:
                            self.logger.exception("Failed to persist sql correction error")
                        return user_msg, log, None, None
                    corr_duration = perf_counter() - t0
                    node_timings.setdefault("08_sql_correct", 0)
                    node_timings["08_sql_correct"] += corr_duration
                    log.append(f"[Node: SQL Generation (Corrected)]\n{sql} ({corr_duration:.3f}s)")
                    if usage:
                        pt = usage.get("prompt_tokens") or 0
                        rt = usage.get("response_tokens") or 0
                        tokens_agg["prompt_tokens"] += pt
                        tokens_agg["response_tokens"] += rt
                        tokens_agg["total_tokens"] += usage.get("total_tokens") or (pt + rt)
                        self.metrics.add_tokens("sql_correct", pt, rt)
                else:
                    self.logger.error("SQL execution failed after retries", exc_info=True)
                    self.metrics.add_query_result(False)
                    try:
                        PersistentMetrics.append_query({
                            "timestamp": pd.Timestamp.now().isoformat(),
                            "user_query": user_query,
                            "generated_sql": sql,
                            "intent": intent,
                            "success": False,
                            "tokens_used": tokens_agg,
                            "response_time_sec": round(perf_counter() - overall_start, 3),
                            "node_timings": node_timings
                        })
                    except Exception:
                        self.logger.exception("Failed to persist failed query")
                    self.last_query = user_query
                    return f"Agent failed to fix the SQL after {max_retries} loop attempts.", log, None, None
            else:
                db_results = results
                success = True
                log.append(f"[Node: SQL Exec Success] Returned {len(db_results)} rows.")
                break

        if not success:
            return "Query execution failed.", log, None, None

        #  Save to conversation memory 
        self.last_query = user_query
        self.last_sql = sql
        self.last_results = db_results
        self.last_df = pd.DataFrame(db_results) if db_results else None

        #  Implicit Chart Generation 
        # Check if the query implies a chart, even if intent was NEW_QUERY
        chart_keywords = ["chart", "graph", "plot", "viz", "display", "trend", "breakdown"]
        if any(kw in user_query.lower() for kw in chart_keywords) and self.last_df is not None:
            log.append("[Node: Implicit Chart Builder] Auto-triggering chart config detection.")
            print_terminal("Implicit Chart Builder")
            t0 = perf_counter()
            try:
                chart_config, usage = self.nodes.chart_config(user_query, self.last_df)
                duration = perf_counter() - t0
                node_timings["09_chart_config"] = duration
                log.append(f"[Node: Implicit Chart] Config: {chart_config} ({duration:.3f}s)")
                # (Optional: track tokens here if needed)
            except Exception:
                self.logger.warning("Implicit chart generation failed, skipping visuals.")

        #  Data Post-Processing and NLG (Next Step) 
        # (Persistence moved to end to capture NLG timing)

        # Node 10: Natural Language Answer
        notify_ui("Generating Answer")
        print_terminal("Natural Answer")
        t0 = perf_counter()
        
        # Prevent the LLM from outputting massive unreadable text walls by limiting its context to 10 rows safely
        import random
        sample_results = db_results
        if isinstance(db_results, list) and len(db_results) > 10:
            sample_results = random.sample(db_results, 10 + attempt*5) # slight expansion on retries
            
        try:
            final_answer, usage = self.nodes.natural_answer(user_query, sample_results, unit_context=unit_context)
        except Exception as e:
            from src.agent.exceptions import AgentError
            if isinstance(e, AgentError):
                user_msg = e.user_message
                self.logger.warning(f"NLG node error: {e.internal_detail}")
            else:
                user_msg = "Internal error generating the answer. Try again later."
                self.logger.exception("Natural language generation failed")
            try:
                PersistentMetrics.append_query({
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "user_query": user_query,
                    "generated_sql": sql,
                    "intent": intent,
                    "success": False,
                    "tokens_used": tokens_agg,
                    "response_time_sec": round(perf_counter() - overall_start, 3),
                    "node_timings": node_timings,
                })
            except Exception:
                self.logger.exception("Failed to persist nlg error")
            self.last_query = user_query
            return user_msg, log, None, None
        duration = perf_counter() - t0
        node_timings["10_nlg"] = duration
        log.append(f"[Node: Natural Answer] Generation complete. ({duration:.3f}s)")
        if usage:
            pt = usage.get("prompt_tokens") or 0
            rt = usage.get("response_tokens") or 0
            tokens_agg["prompt_tokens"] += pt
            tokens_agg["response_tokens"] += rt
            tokens_agg["total_tokens"] += usage.get("total_tokens") or (pt + rt)
            self.metrics.add_tokens("nlg", pt, rt)

        # FINAL PERSISTENCE POINT  Captures 100% of node timings
        try:
            total_time_sec = round(perf_counter() - overall_start, 3)
            PersistentMetrics.append_query({
                "timestamp": pd.Timestamp.now().isoformat(),
                "user_query": user_query,
                "generated_sql": sql,
                "intent": intent,
                "success": True,
                "tokens_used": tokens_agg,
                "response_time_sec": total_time_sec,
                "node_timings": node_timings
            })
            self.metrics.add_query_result(True)
            self.logger.info("[Analytics] Persisted 100% complete query state to history.")
        except Exception:
            self.logger.exception("Failed to persist query history")

        # ── Automated Learning Loop (Background) ────────────────
        # If the query was successfully corrected after a failure,
        # learn from it in parallel to avoid user lag.
        try:
            if node_timings.get("08_sql_correct", 0) > 0 and success:
                # Up 3 directories: src/agent/workflow.py -> /data/few_shot_examples.json
                fs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "few_shot_examples.json")
                
                # Use a background thread for zero-lag learning
                learning_thread = threading.Thread(
                    target=self.vs.learn_from_correction,
                    args=(user_query, sql, "Auto-Corrected execution", fs_path)
                )
                learning_thread.daemon = True # Don't block exit
                learning_thread.start()
                self.logger.info("[Learning] Background learning loop triggered.")
        except Exception as e:
            self.logger.warning(f"[Learning] Background trigger failed: {e}")

        return final_answer, log, chart_config, self.last_df
