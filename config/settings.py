import os 
from pathlib import Path 
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent 
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = DATA_DIR / "temp"
OUTPUT_DIR = DATA_DIR / "output"

for _d in [DATA_DIR , TEMP_DIR, OUTPUT_DIR]:
    _d.mkdir(parents=True , exist_ok=True)

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Child chunk size — small for retrieval precision; parent (full page) goes to LLM
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Search :how many child chunks to retrieve before expanding to parent pages
TOP_K_RETRIEVAL = 8
RRF_K = 60

# Confidence thresholds
HIGH_CONFIDENCE = 0.80
LOW_CONFIDENCE = 0.50

# OCR DPI for PDF conversion
PDF_DPI = 200

OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384
CHUNK_SIZE        = 600    # Small child chunks for retrieval precision
CHUNK_OVERLAP     = 100    # Context preservation across boundaries
TOP_K_RETRIEVAL   = 8      # Child chunks to retrieve per field
RRF_K             = 60     # Reciprocal rank fusion constant
HIGH_CONFIDENCE   = 0.80   # Green threshold
LOW_CONFIDENCE    = 0.50   # Red threshold
PDF_DPI           = 200    # PDF rasterization resolution
TEMP_DIR =""

EXTRACT_FIELDS = [
    "party_a_legal_name",
    "party_b_legal_name",
    "start_date",
    "end_date",
    "payment_conditions",
    "price_details",
]
