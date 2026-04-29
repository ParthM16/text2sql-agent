import os
import json
import time
import pandas as pd
from dotenv import load_dotenv

# Load environment before anything else
load_dotenv()
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.db.helper import DBHelper
from src.retrieval.vector_store import VectorStore
from src.agent.workflow import TextToSQLAgent

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
EVAL_FILE = os.path.join(DATA_DIR, "eval_set.json")
EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "export")
os.makedirs(EXPORT_DIR, exist_ok=True)

def compare_dataframes(df_gold: pd.DataFrame, df_pred: pd.DataFrame) -> tuple[bool, str]:
    if df_pred is None and df_gold is None:
        return True, "Both returned no data."
    if df_pred is None:
        return False, "Predicted returned no data."
    if df_gold is None:
        return False, "Gold returned no data, but predicted returned data."
    
    if df_gold.shape != df_pred.shape:
        return False, f"Shape mismatch: Gold {df_gold.shape} vs Pred {df_pred.shape}"
        
    try:
        df_gold = df_gold.sort_values(by=list(df_gold.columns)).reset_index(drop=True)
        df_pred = df_pred.sort_values(by=list(df_pred.columns)).reset_index(drop=True)
    except Exception:
        pass

    try:
        gold_vals = df_gold.to_numpy()
        pred_vals = df_pred.to_numpy()
        
        import numpy as np
        
        def safe_compare(v1, v2):
            if pd.isna(v1) and pd.isna(v2): return True
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                return np.isclose(float(v1), float(v2), rtol=1e-3, atol=1e-3)
            return str(v1).strip() == str(v2).strip()
            
        for i in range(gold_vals.shape[0]):
            for j in range(gold_vals.shape[1]):
                if not safe_compare(gold_vals[i,j], pred_vals[i,j]):
                    return False, f"Value mismatch at row {i}: Gold '{gold_vals[i,j]}' vs Pred '{pred_vals[i,j]}'"
                    
        return True, "Perfect Match"
        
    except Exception as e:
        return False, f"Comparison error: {str(e)}"

def run_evals():
    print("[INIT] Initializing Evaluation Framework...")
    try:
        db_helper = DBHelper()
        vector_store = VectorStore()
        agent = TextToSQLAgent(db_helper, vector_store)
    except Exception as e:
        print(f"[ERROR] Failed to initialize agent components: {e}")
        return

    if not os.path.exists(EVAL_FILE):
        print(f"[ERROR] Eval file not found at {EVAL_FILE}")
        return

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    print(f"[DATA] Loaded {len(eval_cases)} evaluation cases.")
    
    results = []
    passed = 0
    
    for idx, case in enumerate(eval_cases):
        print(f"\n--- Running Case {idx+1}/{len(eval_cases)} ---")
        question = case["question"]
        gold_sql = case["gold_sql"]
        difficulty = case["difficulty"]
        print(f"Q: {question}")
        
        try:
            gold_rows, err = db_helper.execute_query(gold_sql)
            if err:
                print(f"[WARN] Gold SQL failed: {err}")
                gold_df = None
            else:
                gold_df = pd.DataFrame(gold_rows)
        except Exception as e:
            gold_df = None
            print(f"[WARN] Gold SQL error: {e}")
            
        agent._unit_cache = {}
        
        start_time = time.time()
        ans, logs, cfg, pred_df = agent.run(question)
        eval_time = round(time.time() - start_time, 2)
        
        pred_sql = "N/A"
        for log in logs:
            if log.startswith("[Node: SQL Generation] Query:"):
                pred_sql = log.replace("[Node: SQL Generation] Query:", "").strip()
            if log.startswith("[Node: SQL Correction] Corrected Query:"):
                pred_sql = log.replace("[Node: SQL Correction] Corrected Query:", "").strip()
        
        is_match, reason = compare_dataframes(gold_df, pred_df)
        
        if is_match:
            print("[PASS] Match confirmed")
            passed += 1
        else:
            print(f"[FAIL] {reason}")
            
        results.append({
            "id": idx + 1,
            "question": question,
            "difficulty": difficulty,
            "category": case.get("category", "N/A"),
            "passed": is_match,
            "failure_reason": reason if not is_match else "",
            "eval_time_sec": eval_time,
            "gold_sql": gold_sql,
            "predicted_sql": pred_sql
        })

    accuracy = (passed / len(eval_cases)) * 100
    print(f"\n==========================================")
    print(f"FINAL EVALUATION SCORE: {passed}/{len(eval_cases)} ({accuracy:.1f}%)")
    print(f"==========================================")

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(EXPORT_DIR, "eval_report.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved CSV Report to: {csv_path}")
    
    md_path = os.path.join(EXPORT_DIR, "eval_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Offline Evaluation Report\n\n")
        f.write(f"**Date Run:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Overall Accuracy:** {accuracy:.1f}% ({passed}/{len(eval_cases)} passed)\n\n")
        
        f.write("## Summary Metrics\n")
        f.write(f"- **Total Test Cases:** {len(eval_cases)}\n")
        f.write(f"- **Average Time per Query:** {df_results['eval_time_sec'].mean():.2f}s\n\n")
        
        f.write("## Failure Analysis\n")
        failures = df_results[~df_results['passed']]
        if failures.empty:
            f.write("Perfect score! No failures detected.\n\n")
        else:
            for _, row in failures.iterrows():
                f.write(f"### Case #{row['id']}: {row['question']}\n")
                f.write(f"- **Difficulty**: {row['difficulty']} | **Category**: {row['category']}\n")
                f.write(f"- **Reason**: {row['failure_reason']}\n")
                f.write(f"**Gold SQL:**\n```sql\n{row['gold_sql']}\n```\n")
                f.write(f"**Predicted SQL:**\n```sql\n{row['predicted_sql']}\n```\n")
                f.write("---\n\n")
                
        f.write("## Full Results Table\n")
        f.write("| ID | Question | Passed | Time (s) | Category |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in df_results.iterrows():
            icon = "PASS" if row['passed'] else "FAIL"
            f.write(f"| {row['id']} | {row['question']} | {icon} | {row['eval_time_sec']} | {row['category']} |\n")

    print(f"Saved Markdown Report to: {md_path}")

if __name__ == "__main__":
    run_evals()
