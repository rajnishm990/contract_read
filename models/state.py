from typing import TypedDict , Dict , List , Any  


class ContractState(TypedDict, total=False):
    file_path:str  
    file_type: str 
    original_filename:str 

    page_image_path: List[str]
    deduplicated_page_indices : List[int]

    raw_text_by_page: Dict[int, str]
    full_text = str 

    chunks: List[str]
    chunk_metadata: List[Dict[str, Any]]
    session_faiss_ready: bool 

    extracted_fileds: Dict[str, Dict[str, Any]]

    excel_output_path: str 
    errors : List[str]

    current_step: str 
    processing_log : List[str]
    prompt_log: List[Dict[str, Any]]


