import os 

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