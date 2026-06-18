from typing  import List , Dict , Any 

from langchain_text_splitters import RecursiveCharacterTextSplitter 

from models.state import ContractState
from services.vectror_store import HybridVectorStore 
from config.settings import CHUNK_OVERLAP, CHUNK_SIZE 

_session_store : HybridVectorStore | None=None # resets for each document .. No need to maintian cross document persistence 


def get_session_store() -> HybridVectorStore:
    return _session_store 

def _find_page(chunk:str,raw_text_by_page:Dict[int , str]) -> int:
    needle = chunk[:60].strip() 
    for pg, pg_text in raw_text_by_page.items():
        if needle in pg_text :
            return pg 
    return 0 

def indexing_node(state:ContractState) ->dict:
    global _session_store 
    log = list(state.get("processing_log"), [])

    text = state.get("full_text", "")
    raw_text_by_page =  state.get("raw_text_by_page", {}) 
    source_name = state.get("original_filename", "contract")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE ,
        chunk_overlap = CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )
    chunks : List[str] = splitter.split_text(text)

    metadata: List[Dict[str, Any]] = []
    for i , chunk in enumerate(chunks):
        pg = _find_page(chunk , raw_text_by_page)
        metadata.append({
            "chunk_id": i, 
            "page": pg ,
            "source": source_name,
            "page_text": raw_text_by_page.get(pg,chunk)
        })
    _session_store = HybridVectorStore()
    _session_store.add_documents(chunks, metadata)
    
    log.append(f"Indexed {len(chunks)} chunks into session FAISS+BM25")
    return {
        **state,
        "chunks": chunks,
        "chunk_metadata": metadata,
        "session_faiss_ready": True,
        "processing_log": log,
        "current_step": "extraction",
    }