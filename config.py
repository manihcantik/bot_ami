# config.py
from pathlib import Path

# ==============================
# KONFIGURASI DASAR PROYEK
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent
SCENARIO_NAME = "S1_chunk300_overlap50_top3"

# Parameter Chunking (untuk ingestion data)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Parameter Retrieval
TOP_K = 3

# ==============================
# KONFIGURASI MODEL & DATABASE
# ==============================
EMBEDDING_MODEL = "BAAI/bge-m3"
DB_PATH = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "docs"

# ==============================
# KONFIGURASI LLM (LM Studio)
# ==============================
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL = "google/gemma-4-e2b"
TEMPERATURE = 0.2
MAX_TOKENS = 4096
TIMEOUT_SECONDS = 180

# ==============================
# KONFIGURASI CHATBOT & LOGIKA
# ==============================
MAX_HISTORY = 3
LOG_FILE = str(PROJECT_ROOT / "chatbot_logs.jsonl")
DATASET_EVAL_FILE = PROJECT_ROOT / "dataset_evaluasi.json"
CSV_INPUT_FILE = PROJECT_ROOT / "data_manual.csv"
RELEVANCE_THRESHOLD = 0.25