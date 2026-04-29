import os
import time
import uuid
import pandas as pd
import logging

logger = logging.getLogger(__name__)
CACHE_DIR = "data/cache"

def save_to_cache(df: pd.DataFrame) -> str:
    """Saves a massive raw DataFrame to disk to conserve active Server RAM."""
    if df is None or df.empty:
        return None
        
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        # Unique filename using timestamp + short UUID
        file_id = f"query_{int(time.time())}_{uuid.uuid4().hex[:6]}.csv"
        filepath = os.path.join(CACHE_DIR, file_id)
        
        # Save to disk
        df.to_csv(filepath, index=False)
        return filepath
    except Exception as e:
        logger.error(f"Failed to cache dataframe to disk: {e}")
        return None

def clean_old_cache(max_age_hours=1):
    """Garbage collector: Deletes cache files older than max_age_hours."""
    if not os.path.exists(CACHE_DIR):
        return
        
    now = time.time()
    deleted_count = 0
    try:
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith(".csv"):
                filepath = os.path.join(CACHE_DIR, filename)
                file_age = now - os.path.getmtime(filepath)
                if file_age > (max_age_hours * 3600):
                    os.remove(filepath)
                    deleted_count += 1
        if deleted_count > 0:
            logger.info(f" Garbage Collector: Cleaned {deleted_count} stale cache files.")
    except Exception as e:
        logger.error(f"Garbage collection failed: {e}")
