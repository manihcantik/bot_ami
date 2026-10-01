# config.py
import os
from pathlib import Path

# ==============================
# KONFIGURASI DASAR PROYEK
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_NAME = os.getenv("RAG_SCENARIO", "default")

# Parameter Chunking (untuk ingestion data)
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))

# Parameter Retrieval
DEFAULT_TOP_K = 5
TOP_K = int(os.getenv("RAG_TOP_K", DEFAULT_TOP_K))

# ==============================
# KONFIGURASI MODEL & DATABASE
# ==============================
EMBEDDING_MODEL = os.getenv(
	"RAG_EMBEDDING_MODEL", str(PROJECT_ROOT / "models" / "bge-m3")
)
EMBEDDING_MODEL_PATH = EMBEDDING_MODEL
SUMBER_DATA_DIR = PROJECT_ROOT / "data" / "sumber_data"
HASIL_CHUNKING_DIR = PROJECT_ROOT / "data" / "chunks"
CHROMA_DB_DIR = Path(
	os.getenv("RAG_CHROMA_DB_DIR", str(PROJECT_ROOT / "vectorstore" / "default"))
)
DB_PATH = str(CHROMA_DB_DIR)
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
LOG_FILE = os.getenv(
	"RAG_LOG_FILE", str(PROJECT_ROOT / "results" / "chatbot_logs.jsonl")
)
GUARDRAIL_LOG_FILE = os.getenv(
	"RAG_GUARDRAIL_LOG_FILE", str(PROJECT_ROOT / "results" / "guardrail_logs.jsonl")
)
DATASET_EVAL_FILE = Path(
	os.getenv(
		"RAG_DATASET_EVAL_FILE",
		str(PROJECT_ROOT / "data" / "questions" / "dataset_evaluasi.json"),
	)
)
CSV_INPUT_FILE = Path(
	os.getenv(
		"RAG_CSV_INPUT_FILE",
		str(PROJECT_ROOT / "data" / "questions" / "questions.csv"),
	)
)
RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.25"))
GUARDRAIL_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "2000"))