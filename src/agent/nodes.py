"""
Agent Nodes  Pure LLM Prompt Functions
========================================
Each method is a self-contained "node" in the agentic graph.
They accept structured inputs and return structured outputs.
The orchestration logic lives in workflow.py.
"""
import os
import pandas as pd
from google import genai

from src.agent.exceptions import LLMError
from src.utils.logger import get_logger


class AgentNodes:
    """Encapsulates all LLM-backed reasoning nodes used by the agent."""

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        project_id = os.getenv("PROJECT_ID", "")
        location = os.getenv("LOCATION", "us-central1")
        api_key = os.getenv("GOOGLE_API_KEY", "")

        if project_id:
            #  Vertex AI Mode 
            # Routes through aiplatform.googleapis.com (Vertex AI API)
            # Requires: gcloud auth application-default login
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            print(f"[INFO] Using Vertex AI API (project: {project_id}, location: {location})")
        elif api_key and api_key != "your_api_key_here":
            #  Gemini API Mode (fallback) 
            # Routes through generativelanguage.googleapis.com
            self.client = genai.Client(api_key=api_key)
            print("[INFO] Using Gemini API (API key)")
        else:
            self.client = None
            print("[WARN] No API credentials found. Set PROJECT_ID or GOOGLE_API_KEY in .env")
        self.model_name = model_name
        self.logger = get_logger(__name__)

    def _generate(self, prompt: str) -> tuple[str, dict]:
        """Centralized LLM call returning (text, usage_dict).

        Wraps SDK calls and converts exceptions into `LLMError`.
        """
        if not self.client:
            raise LLMError("AI client not configured.", internal_detail="No API credentials")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            # Extract text
            text = getattr(response, "text", None)
            if text is None:
                # SDK sometimes returns candidates
                candidates = getattr(response, "candidates", None)
                if candidates and len(candidates) > 0:
                    text = getattr(candidates[0], "content", "")
            text = text.strip() if text else ""

            # Extract usage metadata if available
            usage = {}
            um = getattr(response, "usage_metadata", None)
            if um:
                usage = {
                    "prompt_tokens": getattr(um, "prompt_token_count", None),
                    "response_tokens": getattr(um, "candidates_token_count", None),
                    "total_tokens": getattr(um, "total_token_count", None),
                }
            return text, usage
        except Exception as e:
            self.logger.exception("LLM call failed")
            raise LLMError("AI model failed. Please try again.", internal_detail=str(e))

    # 
    # Node: Intent Detection
    # 
    def intent_detection(self, user_query: str, last_query: str = None,
                         last_sql: str = None, last_results: list = None,
                         last_df: pd.DataFrame = None) -> tuple[str, dict]:
        """Classifies user intent: NEW_QUERY, CHART_REQUEST, FOLLOW_UP, or EXPLORATORY."""
        history_context = ""
        if last_query:
            cols = list(last_df.columns) if last_df is not None else "N/A"
            rows = len(last_results) if last_results else 0
            history_context = f"""
            The previous user question was: "{last_query}"
            The previous SQL executed was: {last_sql}
            The previous result had {rows} rows with columns: {cols}
            """

        prompt = f"""
        You are a smart intent classifier for a Text-to-SQL chatbot.

        {history_context}

        The new user message is: "{user_query}"

        Classify the intent into EXACTLY one of these four categories:
        - EXPLORATORY     user is asking what they CAN do, requesting suggestions, exploring capabilities, or asking for help (e.g., "what analysis can I do?", "what data do you have?", "help me explore", "what can you tell me?", "suggest something", "what insights are available?").
        - CHART_REQUEST   ONLY if the user wants to visualize the EXACT data already seen (e.g., "Chart this", "Make it a graph", "Visualize that data").
        - FOLLOW_UP       user is asking a follow-up about previous results OR wanting new filters/aggregations (e.g. "now show it for 2019", "break this down by region").
        - NEW_QUERY       user is asking a brand new database question, even if they ask for it in a chart.

        Output ONLY the category name. Nothing else.
        """
        text, usage = self._generate(prompt)
        intent = text.upper().replace(" ", "_")
        
        # Guardrails
        if "EXPLOR" in intent:
            return "EXPLORATORY", usage
        if "CHART" in intent:
            return "CHART_REQUEST", usage
        if "FOLLOW" in intent:
            return "FOLLOW_UP", usage
        return "NEW_QUERY", usage

    # 
    # Node: Exploratory Answer (Schema-Only, No SQL)
    # 
    def exploratory_answer(self, user_query: str, schema: str) -> tuple[str, dict]:
        """Generates analysis suggestions based on schema alone  no SQL executed."""
        prompt = f"""
        You are a senior Data Analyst. The user asked: "{user_query}"

        Here is the database schema they have access to:
        {schema}

        Based ONLY on the schema above, suggest 5-7 specific, actionable analyses the user can perform.
        For each suggestion:
        - Give it a short bold title
        - Explain what question it answers in 1 sentence
        - Mention which tables/columns would be used

        Format as a clean numbered list with markdown. Keep it concise and professional.
        Do NOT generate any SQL. Do NOT run any queries. Just suggest what's possible.
        """
        text, usage = self._generate(prompt)
        return text, usage

    # 
    # Node: Chart Configuration
    # 
    def chart_config(self, user_query: str, df: pd.DataFrame) -> tuple[dict, dict]:
        """Uses LLM to determine the best chart type and axes from the dataframe."""
        cols = list(df.columns)
        sample = df.head(6).to_string()
        prompt = f"""
        The user asked: "{user_query}"
        The data has columns: {cols}
        Sample rows:
        {sample}

        Pick the best chart type and columns for a chart.

        IMPORTANT - Detect the data format:
        FORMAT A (Wide data): Multiple numeric columns to compare (e.g., UniqueCustomers, NonUniqueCustomers).
                              List them all as Y_COLUMNS.
        FORMAT B (Long data): ONE value column and ONE category column that splits data into groups.
                              Example: customer_type has values New/Returning, customer_count has the numbers.
                              Set Y_COLUMNS to the value column AND set COLOR_COLUMN to the category column.

        Respond in EXACTLY this format (no extra text):
        CHART_TYPE: bar or line or area
        X_COLUMN: <column_name>
        Y_COLUMNS: <column_name1>, <column_name2>
        COLOR_COLUMN: <column_name or NONE>
        TITLE: <short chart title>
        """
        text, usage = self._generate(prompt)

        config = {"chart_type": "bar", "x": cols[0], "y": [cols[-1]], "title": "Chart", "color": None}
        for line in text.split("\n"):
            line = line.strip()
            if not line: continue
            
            try:
                parts = line.split(":", 1)
                if len(parts) < 2: continue
                key, val = parts[0].strip().upper(), parts[1].strip()
                
                if key == "CHART_TYPE":
                    if val.lower() in ("bar", "line", "area"):
                        config["chart_type"] = val.lower()
                elif key == "X_COLUMN":
                    if val in cols: config["x"] = val
                elif key == "Y_COLUMNS":
                    raw_vals = [v.strip() for v in val.split(",")]
                    valid_vals = [v for v in raw_vals if v in cols]
                    if valid_vals: config["y"] = valid_vals
                elif key == "COLOR_COLUMN":
                    if val.upper() != "NONE" and val in cols:
                        config["color"] = val
                elif key == "TITLE":
                    config["title"] = val
            except Exception:
                continue
        return config, usage

    # 
    # Node: Table Selection
    # 
    def table_selection(self, schema_string: str, query: str) -> tuple[str, dict]:
        """Selects relevant tables from the schema to reduce prompt size."""
        prompt = f"""
        Given the following database schema:
        {schema_string}
        
        And the user query: "{query}"
        
        Return exactly a comma-separated list of ONLY the relevant table names required to answer the query. Do not include any other text.
        """
        text, usage = self._generate(prompt)
        return text.replace('`', ''), usage

    # 
    # Node: SQL Generation
    # 
    def sql_generation(self, query: str, context_schema: str,
                       few_shots: list, corrected_attributes: list,
                       db_mode: str = "sqlserver",
                       unit_conversion_rules: str = None) -> tuple[str, dict]:
        """Generates the SQL query using the correct dialect for the active DB mode."""
        few_shot_str = "\n".join(
            [f"Query: {ex['query']}\nSQL: {ex['sql']}" for ex in few_shots]
        )
        attr_str = "\n".join(
            [f"Use '{m['val']}' for column '{m['col']}' instead of what user provided."
             for m in corrected_attributes if m]
        )

        # Dialect-aware instruction
        if db_mode == "sqlite":
            dialect_instruction = (
                "Generate a syntactically correct **SQLite** query. "
                "Use LIMIT (not TOP), use || for string concatenation, "
                "and do NOT use square brackets for column names."
            )
        else:
            dialect_instruction = (
                "Generate a syntactically correct **MS SQL Server (T-SQL)** query. "
                "Use TOP (not LIMIT), use + for string concatenation, "
                "and wrap column names in square brackets if they contain spaces."
            )

        prompt = f"""
        You are an expert SQL Developer for Walmart / Retail Data.
        Given the following Schema context (only relevant tables):
        {context_schema}
        
        Here are some examples of previous identical queries:
        {few_shot_str}
        
        Attribute typo corrections to apply:
        {attr_str}

        {dialect_instruction}
        
        {unit_conversion_rules if unit_conversion_rules else ""}

        PERFORMANCE RULES (follow strictly):
        - Do NOT JOIN with other tables if the required data already exists in a single table.
          For example, if a table has an 'order_date' column, extract the year directly
          using YEAR(order_date) or STRFTIME('%Y', order_date) instead of JOINing with a date dimension table.
        - Only use JOINs when data genuinely lives in separate tables (e.g., customer name is in a customers table).
        - Prefer COUNT(*) over COUNT(column_name) when counting rows, unless NULLs matter.
        - Avoid subqueries when a simple GROUP BY achieves the same result.
        - **LATE-JOIN OPTIMIZATION (HIGH PERFORMANCE)**: For high-volume tables (like transactions), always strive to perform JOINS as late as possible. Aggregate/filter the transaction data in a CTE first to reduce the row count, then JOIN with metadata tables (like `customers` or `markets`) only in the final step. This prevents the database from joining millions of unnecessary rows.
        - **CLEAN AGGREGATIONS (MANDATORY  HIGHEST PRIORITY)**:
          When using GROUP BY, SUM, AVG, COUNT, or any aggregation function:
           NEVER include currency, unit, or symbol columns (e.g., MAX(currency), MIN(unit)).
           NEVER add currency/unit columns to GROUP BY.
           NEVER use a CASE statement to label the currency in aggregated output.
           Only output the numeric metric itself. Unit discovery is handled separately by the system.
          Example of FORBIDDEN SQL: SELECT YEAR(date), SUM(amount), MAX(currency) FROM t GROUP BY YEAR(date)
          Example of CORRECT SQL:   SELECT YEAR(date), SUM(amount) FROM t GROUP BY YEAR(date)

        RETAIL DOMAIN RULES (critical for correct business logic):
        - "Unique customer" or "New customer" = a customer who appears for the FIRST TIME EVER in that year.
          They did NOT exist in any previous year's data. Find them by comparing each customer's
          MIN(year) against the current year.
        - "Repeated customer" or "Returning customer" = a customer who ALREADY appeared in a PREVIOUS year.
          Their first-ever year (MIN year) is earlier than the current year being analyzed.
        - Do NOT confuse "unique/new" with "purchased once" or "distinct count".
          When users say "unique vs repeated customers year-wise", they mean:
            New = first_purchase_year == current_year
            Returning = first_purchase_year < current_year
        - "Retention" = percentage of previous year's customers who came back this year.
        - "Churn" = percentage of previous year's customers who did NOT come back this year.
        - "Profit Margin" (per transaction) = (sales_amount - cost_price).
        - "Margin Contribution" (per customer/year) = SUM((sales_amount - cost_price) * sales_qty). 
          This is the true measure of a customer's value as it weights margin by quantity.
        - **CODE-FIRST RULE**: NEVER use the pre-calculated `profit_margin` or `profit_margin_percentage` columns from the database. ALWAYS calculate these metrics from raw components (sales_amount, cost_price, sales_qty) in your SQL code.
        - Always use these formulas when a user asks for 'highest margin' or 'best contributor'.
        
        CONTEXT COLUMN RULE:
        - If the query is a **non-aggregated** query (no GROUP BY, no SUM/AVG), and the schema has context columns (currency, unit, symbol), include them in SELECT for transparency.
        - If the query **uses aggregation** (GROUP BY, SUM, AVG, COUNT), **do NOT include any context columns**. The CLEAN AGGREGATION rule above takes absolute priority.
        
        Generate a query to answer this user request:
        "{query}"
        
        Output ONLY the raw SQL query, no markdown blocks, no explanation.
        
        SMART SCHEMA DETECTION:
        If you realize the user is asking for a column/data point that COMPLETELY DOES NOT EXIST in the schema provided above (e.g. they ask for 'department' but there is no such field), DO NOT hallucinate a query.
        Instead, output EXACTLY a string starting with "NOT_FOUND:" followed by a smart UI message telling them it's missing, and suggesting 2 or 3 of the closest available columns/tables to help them.
        Example: "NOT_FOUND: I couldn't find a 'department' field in our sales database. However, I did find 'Market Code' and 'Customer Code'. Would you like me to check any of those?"
        """
        text, usage = self._generate(prompt)
        
        if text.strip().startswith("NOT_FOUND:"):
            return text.strip(), usage
            
        return text.replace('```sql', '').replace('```', ''), usage

    # 
    # Node: SQL Correction
    # 
    def sql_correction(self, query: str, bad_sql: str,
                       error_msg: str, schema: str,
                       db_mode: str = "sqlserver") -> tuple[str, dict]:
        """Corrects a failed SQL query using the correct dialect."""
        dialect = "SQLite" if db_mode == "sqlite" else "MS SQL Server (T-SQL)"

        prompt = f"""
        The following {dialect} query failed with an error. 
        User Request: {query}
        Bad SQL: {bad_sql}
        Error: {error_msg}
        Schema: {schema}
        
        Fix the SQL query for {dialect}. Output ONLY the raw SQL, no markdown, no explanation.
        """
        text, usage = self._generate(prompt)
        return text.replace('```sql', '').replace('```', ''), usage



    # 
    # Node: Natural Language Answer
    # 
    def natural_answer(self, query: str, db_results: list, 
                       unit_context: dict = None) -> tuple[str, dict]:
        """Generates a natural language response to the user."""
        str_results = str(db_results)[:15000]
        
        collision_warning = ""
        if unit_context and unit_context.get("collision_detected"):
            msg = unit_context.get("caution_message", "Inconsistent units detected.")
            factor_note = ""
            if unit_context.get("live_web_factor"):
                factor_note = f"\n> **CONVERSION APPLIED**: {unit_context.get('live_web_factor')}"
                
            collision_warning = f"""
            >  **[!CAUTION] DATA INTEGRITY ALERT**: {msg}
            > The units found are: {', '.join(unit_context.get('units_found', []))}. {factor_note}
            > To ensure 100% accuracy, please consider adding a 'unit_master' conversion table to your database.
            """

        prompt = f"""
        The user asked: "{query}"
        The database returned the following JSON rows:
        {str_results}

        {collision_warning}

        Important instructions for the response:
        - Provide the exact information requested by the user. 
        - If a CAUTION block is provided above, you MUST lead your answer with it and EXPLICITLY mention BOTH the conversion rate used AND the source of that information (e.g. "1 [Unit A] = [Factor] [Unit B], Source: [Source Name]") in the very first sentence.
        - If returning a single record or tabular data, FORMAT IT BEAUTIFULLY using Markdown formatting. 
        - Start with a direct human-friendly statement summarizing the main finding.
        - **CONTEXT-AWARE FORMATTING (THE HIERARCHY OF TRUST)**: 
          Always format numeric values with their correct units following this exact priority:
          1. **Primary: Database Context (THE GROUND TRUTH)**. 
          2. **Secondary: Semantic Descriptions**.
          3. **Fallback: Ambiguity Disclaimer**.
        - Use bolding for key metrics, and display record fields as clean bullet points or a Markdown table.
        - Do not include SQL code. Output ONLY the final markdown answer.
        - IMPORTANT: This system has a built-in visualization engine. Do NOT include any Python code, 
        Matplotlib, Plotly, or any visualization libraries in your output. Simply answer the question 
        summarizing the data. If the user asked for a chart, do NOT explain how to create one; just 
        state that the chart is being displayed.
        """
        text, usage = self._generate(prompt)
        return text, usage
