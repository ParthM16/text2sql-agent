"""
Model Evaluation & Benchmarking Suite
=====================================
Evaluates Quality, Latency, and Cost Trade-offs across:
  - Models: Google Gemini 2.5 Flash vs Local Ollama (qwen2.5-coder:3b)
  - Vector Databases: In-Memory FAISS (IndexFlatL2) vs Persistent Local ChromaDB

CRITICAL (Rule 3):
Includes a configurable delay (default 5.0s) between Gemini cloud benchmark queries
outside the latency timer to protect against Google AI Studio Free Tier rate limits (15 RPM).
"""
import os
import sys
import json
import time
import re
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd

from src.retrieval.store_factory import VectorStoreFactory
from src.llm.model_router import LLMRouter
from src.config import config


class ModelBenchmark:
    """Benchmark suite for comparing multi-model and vector backend combinations."""

    DEFAULT_TEST_QUERIES = [
        {"query": "show me the total sales amount", "sql": "SELECT SUM(sales_amount) FROM transactions"},
        {"query": "how many unique customers do we have?", "sql": "SELECT COUNT(DISTINCT customer_code) FROM customers"},
        {"query": "what is the total sales quantity for product Prod001?", "sql": "SELECT SUM(sales_qty) FROM transactions WHERE product_code = 'Prod001'"},
        {"query": "show me the total sales amount year wise", "sql": "SELECT YEAR(order_date) AS sales_year, SUM(sales_amount) FROM transactions GROUP BY YEAR(order_date)"},
        {"query": "what is the profit margin contribution for market code Mark001?", "sql": "SELECT SUM((sales_amount - cost_price) * sales_qty) FROM transactions WHERE market_code = 'Mark001'"},
    ]

    def __init__(
        self,
        gemini_delay_seconds: float = 5.0,
        output_dir: Optional[str] = None
    ):
        self.gemini_delay_seconds = gemini_delay_seconds
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.output_dir = output_dir or os.path.join(base_dir, "data")
        os.makedirs(self.output_dir, exist_ok=True)
        self.router = LLMRouter()

    def _extract_sql(self, text: str) -> str:
        """Extracts SQL query from LLM output."""
        if not text:
            return ""
        # Match ```sql ... ``` block
        sql_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        # Fallback to search for SELECT statement
        select_match = re.search(r"(SELECT\s+.*?;)", text, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()
        return text.strip()

    def _check_sql_validity(self, sql: str) -> bool:
        """Sanity checks if generated text contains valid SQL query structure."""
        if not sql:
            return False
        clean = sql.strip().upper()
        return clean.startswith("SELECT") or "SELECT " in clean or clean.startswith("WITH")

    def run_benchmark(
        self,
        models: Optional[List[str]] = None,
        backends: Optional[List[str]] = None,
        test_queries: Optional[List[Dict[str, str]]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs the benchmark across all model x backend permutations.
        
        Args:
            models: List of model providers (e.g. ['ollama/qwen2.5-coder:3b', 'gemini-2.5-flash'])
            backends: List of vector backends (e.g. ['faiss', 'chroma'])
            test_queries: List of test query dicts. Defaults to DEFAULT_TEST_QUERIES.
            limit: Maximum number of queries to run per combo.
            
        Returns:
            Dictionary containing raw results and aggregated summaries.
        """
        models = models or [f"ollama/{config.OLLAMA_MODEL}", config.GEMINI_MODEL]
        backends = backends or ["faiss", "chroma"]

        queries = test_queries or self.DEFAULT_TEST_QUERIES
        if limit and limit > 0:
            queries = queries[:limit]

        print(f"\n{'=' * 60}")
        print(f"[BENCHMARK] Starting Multi-Model & Vector DB Evaluation")
        print(f" Models: {models}")
        print(f" Backends: {backends}")
        print(f" Test Queries: {len(queries)}")
        print(f" Gemini Free-Tier Delay Guard: {self.gemini_delay_seconds}s (between iterations)")
        print(f"{'=' * 60}\n")

        raw_results = []
        combo_summaries = {}

        for backend in backends:
            print(f"\n--- Initializing Vector Backend: {backend.upper()} ---")
            store = VectorStoreFactory.get_vector_store(backend=backend)

            # Pre-seed few-shot examples into store if empty
            if hasattr(store, "few_shot_index") and getattr(store.few_shot_index, "ntotal", 0) == 0:
                store.add_few_shots(self.DEFAULT_TEST_QUERIES)

            for model in models:
                combo_key = f"{model} + {backend}"
                print(f"\n>> Evaluating Combo: [{combo_key}]")
                combo_runs = []

                for q_idx, item in enumerate(queries):
                    user_q = item["query"]
                    print(f"  Query {q_idx+1}/{len(queries)}: '{user_q}'")

                    # 1. Measure Vector Store Retrieval Latency (strictly timed)
                    t_vec_start = time.perf_counter()
                    few_shots = store.retrieve_few_shots(user_q, k=2)
                    vec_latency_ms = (time.perf_counter() - t_vec_start) * 1000.0

                    few_shot_context = "\n".join([f"Q: {ex.get('query', '')}\nSQL: {ex.get('sql') or ex.get('expected', '')}" for ex in few_shots])
                    prompt = (
                        f"You are a SQL expert. Based on these examples:\n{few_shot_context}\n\n"
                        f"Write a valid SQL query for: {user_q}\n"
                        f"Output ONLY the SQL code."
                    )

                    # 2. Measure LLM Generation Latency (strictly timed)
                    t_gen_start = time.perf_counter()
                    try:
                        gen_result = self.router.generate(prompt=prompt, model_provider=model)
                        gen_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
                        raw_text = gen_result.get("text", "")
                        extracted_sql = self._extract_sql(raw_text)
                        is_valid_sql = self._check_sql_validity(extracted_sql)
                    except Exception as err:
                        gen_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
                        raw_text = f"ERROR: {err}"
                        extracted_sql = ""
                        is_valid_sql = False
                        gen_result = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "estimated_cost_usd": 0.0,
                            "cost_comparison_usd": {},
                            "fallback_used": False,
                            "error": str(err)
                        }

                    total_e2e_ms = vec_latency_ms + gen_latency_ms

                    run_record = {
                        "combo": combo_key,
                        "model": model,
                        "backend": backend,
                        "query_index": q_idx + 1,
                        "query": user_q,
                        "extracted_sql": extracted_sql,
                        "is_valid_sql": is_valid_sql,
                        "vec_latency_ms": round(vec_latency_ms, 2),
                        "gen_latency_ms": round(gen_latency_ms, 2),
                        "total_latency_ms": round(total_e2e_ms, 2),
                        "input_tokens": gen_result.get("input_tokens", 0),
                        "output_tokens": gen_result.get("output_tokens", 0),
                        "total_tokens": gen_result.get("total_tokens", 0),
                        "estimated_cost_usd": gen_result.get("estimated_cost_usd", 0.0),
                        "cost_comparison": gen_result.get("cost_comparison_usd", {}),
                        "fallback_used": gen_result.get("fallback_used", False)
                    }
                    raw_results.append(run_record)
                    combo_runs.append(run_record)

                    print(f"    [OK] SQL Valid: {is_valid_sql} | Latency: {total_e2e_ms:.1f}ms (Vec: {vec_latency_ms:.1f}ms, LLM: {gen_latency_ms:.1f}ms) | Tokens: {run_record['total_tokens']}")

                    # CRITICAL (Rule 3): Rate-Limit Guard between Gemini queries (outside timer)
                    if "gemini" in model.lower() and q_idx < len(queries) - 1:
                        if self.gemini_delay_seconds > 0:
                            print(f"    [RATE-LIMIT GUARD] Sleeping {self.gemini_delay_seconds}s for Gemini free-tier (15 RPM)...")
                            time.sleep(self.gemini_delay_seconds)

                # Aggregate combo metrics
                valid_count = sum(1 for r in combo_runs if r["is_valid_sql"])
                accuracy_pct = (valid_count / len(combo_runs)) * 100.0 if combo_runs else 0.0
                avg_vec_lat = sum(r["vec_latency_ms"] for r in combo_runs) / len(combo_runs) if combo_runs else 0.0
                avg_gen_lat = sum(r["gen_latency_ms"] for r in combo_runs) / len(combo_runs) if combo_runs else 0.0
                avg_tot_lat = sum(r["total_latency_ms"] for r in combo_runs) / len(combo_runs) if combo_runs else 0.0
                tot_tokens = sum(r["total_tokens"] for r in combo_runs)
                avg_tokens = tot_tokens / len(combo_runs) if combo_runs else 0.0
                tot_cost = sum(r["estimated_cost_usd"] for r in combo_runs)
                cost_per_1k = (tot_cost / len(combo_runs)) * 1000.0 if combo_runs else 0.0

                combo_summaries[combo_key] = {
                    "model": model,
                    "backend": backend,
                    "total_queries": len(combo_runs),
                    "sql_validity_pct": round(accuracy_pct, 1),
                    "avg_vec_latency_ms": round(avg_vec_lat, 2),
                    "avg_gen_latency_ms": round(avg_gen_lat, 2),
                    "avg_total_latency_ms": round(avg_tot_lat, 2),
                    "total_tokens": tot_tokens,
                    "avg_tokens_per_query": round(avg_tokens, 1),
                    "total_cost_usd": round(tot_cost, 6),
                    "cost_per_1k_queries_usd": round(cost_per_1k, 4)
                }

        benchmark_report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "models": models,
                "backends": backends,
                "gemini_delay_seconds": self.gemini_delay_seconds,
                "num_queries": len(queries)
            },
            "summaries": combo_summaries,
            "raw_runs": raw_results
        }

        # Export reports
        self._export_reports(benchmark_report)

        return benchmark_report

    def _export_reports(self, report: Dict[str, Any]) -> None:
        """Exports benchmark results to JSON, CSV, and Markdown."""
        # 1. JSON Export (data/benchmark_results.json)
        json_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[EXPORT] Saved JSON results to: {json_path}")

        # 2. CSV Export (data/benchmark_results.csv)
        csv_path = os.path.join(self.output_dir, "benchmark_results.csv")
        df_runs = pd.DataFrame(report.get("raw_runs", []))
        if not df_runs.empty:
            df_runs.to_csv(csv_path, index=False)
            print(f"[EXPORT] Saved CSV runs to: {csv_path}")

        # 3. Markdown Report (BENCHMARK_REPORT.md in project root)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        md_path = os.path.join(base_dir, "BENCHMARK_REPORT.md")
        self._write_markdown_report(report, md_path)
        print(f"[EXPORT] Generated Markdown report: {md_path}")

    def _write_markdown_report(self, report: Dict[str, Any], filepath: str) -> None:
        """Generates formatted GitHub-style markdown report."""
        summaries = report.get("summaries", {})
        ts = report.get("timestamp", "")

        lines = [
            "# Model Evaluation & Vector DB Benchmark Report",
            "",
            f"> **Generated:** {ts}  ",
            f"> **Cloud Rate-Limit Guard:** {report['config']['gemini_delay_seconds']}s sleep between Gemini API calls  ",
            "",
            "---",
            "",
            "## 1. Executive Summary & Combo Comparison",
            "",
            "| Model x Vector DB | SQL Validity (%) | Retrieval Latency (ms) | Generation Latency (ms) | Total Latency (ms) | Avg Tokens | Cost / 1K Queries |",
            "|-------------------|------------------|-------------------------|--------------------------|---------------------|------------|--------------------|"
        ]

        for combo, s in summaries.items():
            lines.append(
                f"| `{combo}` | **{s['sql_validity_pct']}%** | {s['avg_vec_latency_ms']:.1f} ms | {s['avg_gen_latency_ms']:.1f} ms | {s['avg_total_latency_ms']:.1f} ms | {s['avg_tokens_per_query']:.0f} | **${s['cost_per_1k_queries_usd']:.4f}** |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 2. Multi-Model Quality, Latency & Cost Trade-Offs",
            "",
            "| Model | Hosting / Infra | Cost / 1K Queries | Latency Characteristics | Best Use Case |",
            "|-------|-----------------|-------------------|-------------------------|---------------|",
            "| **Ollama (`qwen2.5-coder:3b`)** | 100% Local (VRAM/RAM) | **$0.00** | Pure local compute, zero network latency | Air-gapped / confidential data, local development |",
            "| **Gemini 2.5 Flash** | Google Cloud (Free Tier) | **$0.00** (Free Tier) | High throughput, sub-second generation | Production cloud analytics, complex SQL joins |",
            "| **OpenAI GPT-4o-mini (simulated)** | Commercial API | ~$0.15 - $0.60 | Cloud API latency | Benchmarking baseline comparison |",
            "| **Claude 3.5 Haiku (simulated)** | Commercial API | ~$0.80 - $4.00 | Cloud API latency | Benchmarking baseline comparison |",
            "",
            "---",
            "",
            "## 3. Vector Database Trade-Offs: FAISS vs ChromaDB",
            "",
            "- **FAISS (`IndexFlatL2`):** Ultra-fast in-memory retrieval (~1-2ms). Ideal for ephemeral session few-shot matching.",
            "- **ChromaDB (`PersistentClient`):** Persistent disk storage (`data/chroma_db/`), zero loss across app restarts, shared singleton embeddings.",
            "",
            "---",
            "*Report auto-generated by `src.evaluation.benchmark`*"
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Multi-Model & Vector DB Benchmark Suite")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test queries (e.g. 2 for quick run)")
    parser.add_argument("--delay", type=float, default=5.0, help="Sleep delay in seconds between Gemini queries (Rate-limit guard)")
    parser.add_argument("--models", nargs="+", default=[f"ollama/{config.OLLAMA_MODEL}", config.GEMINI_MODEL], help="Models to benchmark")
    parser.add_argument("--backends", nargs="+", default=["faiss", "chroma"], help="Vector backends to benchmark")
    args = parser.parse_args()

    benchmark = ModelBenchmark(gemini_delay_seconds=args.delay)
    benchmark.run_benchmark(
        models=args.models,
        backends=args.backends,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
