"""
Pre-download Script for AI Models
=================================
Run this script once before starting the Streamlit app.
It will force HuggingFace to download and cache the models to disk (~90MB),
meaning the Streamlit app will load them instantly in the future instead of downloading.

Usage:
  python scripts/download_models.py
"""
import os
import sys

# Force the script to cache into the local project models directory
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

print("📦 Initializing model download...")
from sentence_transformers import SentenceTransformer

# The model name MUST match what is used in src/retrieval/vector_store.py
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"⏳ Downloading and caching '{MODEL_NAME}' to local disk...")
print("If you have a slow internet connection, this may take a few minutes...")

try:
    # Instantiating the model forces downloading the weights into the local ~/.cache directory
    model = SentenceTransformer(MODEL_NAME)
    print("\n✅ Success! All necessary ML model files have been downloaded and cached.")
    print("You can now run 'streamlit run app.py' and it will boot much faster without downloading.")
except Exception as e:
    print(f"\n❌ Error downloading model: {e}")
    sys.exit(1)
