import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

# Paths for persistent storage
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
QUERY_HISTORY_FILE = os.path.join(DATA_DIR, "query_history.json")
METRICS_HISTORY_FILE = os.path.join(DATA_DIR, "metrics_history.json")

class SessionMetrics:
    """In-memory tracking of metrics for a single Streamlit session."""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        
        # Token tracking
        self.tokens_total = 0
        self.tokens_by_node = {
            "intent": 0,
            "table_select": 0,
            "sql_gen": 0,
            "sql_correct": 0,
            "nlg": 0,
            "chart": 0
        }
        
        # Timing tracking
        self.time_by_node = {} # e.g. {"sql_gen": [1.2, 1.5, 0.9]}
        
        # Errors
        self.errors = [] # List of {"timestamp": str, "node": str, "message": str}
        
    def add_query_result(self, is_success: bool):
        self.total_queries += 1
        if is_success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
            
    def add_tokens(self, node_id: str, prompt_tokens: int, response_tokens: int):
        total = prompt_tokens + response_tokens
        self.tokens_total += total
        if node_id in self.tokens_by_node:
            self.tokens_by_node[node_id] += total
            
    def add_timing(self, node_id: str, duration_sec: float):
        if node_id not in self.time_by_node:
            self.time_by_node[node_id] = []
        self.time_by_node[node_id].append(duration_sec)
        
    def log_error(self, node_id: str, message: str):
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "node": node_id,
            "message": message
        })

    def get_avg_response_time(self) -> float:
        """Returns avg total processing time across all queries in this session."""
        all_times = []
        # Not perfect, treating sum of node times as total time
        total_time_sum = 0
        for times in self.time_by_node.values():
            total_time_sum += sum(times)
            
        if self.total_queries == 0:
            return 0.0
        return total_time_sum / self.total_queries

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def load_json_safely(filepath: str, default_val: Any) -> Any:
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default_val
    return default_val

def save_json_safely(filepath: str, data: Any):
    ensure_data_dir()
    temp_file = filepath + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.replace(temp_file, filepath)

class PersistentMetrics:
    """Handles read/write of long-term history data."""
    
    @staticmethod
    def get_query_history() -> List[Dict]:
        return load_json_safely(QUERY_HISTORY_FILE, [])
        
    @staticmethod
    def append_query(query_data: Dict):
        history = PersistentMetrics.get_query_history()
        # Ensure query_data has a timestamp
        if "timestamp" not in query_data:
            query_data["timestamp"] = datetime.now().isoformat()
        history.append(query_data)
        save_json_safely(QUERY_HISTORY_FILE, history)
        
    @staticmethod
    def clear_query_history():
        save_json_safely(QUERY_HISTORY_FILE, [])
        
    @staticmethod
    def update_daily_metrics(session: SessionMetrics):
        """Aggregate session metrics into daily buckets."""
        if session.total_queries == 0:
            return
            
        history = load_json_safely(METRICS_HISTORY_FILE, {})
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in history:
            history[today] = {
                "total_queries": 0,
                "successful_queries": 0,
                "total_tokens": 0
            }
            
        history[today]["total_queries"] += session.total_queries
        history[today]["successful_queries"] += session.successful_queries
        history[today]["total_tokens"] += session.tokens_total
        
        save_json_safely(METRICS_HISTORY_FILE, history)
