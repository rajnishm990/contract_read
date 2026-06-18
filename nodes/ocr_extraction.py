from typing import Dict  

from paddleocr import PaddleOCR 

from models.state import ContractState 
from services.llm import LLMService 
from utils.file_utils import extract_native_pd_text  
from utils import progress 

#lazy init singleton class so that model only loads once per process 
_ocr_engine: PaddleOCR | None = None 

def _get_ocr() -> PaddleOCR:
    global _ocr_engine 
    if _ocr_engine is None:
        # PP-OCRv4 mobile models are ~4x faster than v5 server on CPU.
        # All rotation/orientation detection disabled — no autorotation.
        _ocr_engine = PaddleOCR(
            use_doc_orientation_classify=False ,
            use_doc_unwarping=False,
            use_textline_orientation=False ,
            text_detection_model_name="PP-OCRv4_mobile_det",
            text_recognition_model_name="PP-OCRv4_mobile_rec",
        )
    return _ocr_engine 

def _ocr_image(path:str) -> str:
    ocr = _get_ocr() 
    try: 
        results = list(ocr.predict(path))
        if not results:
            return ""  
        res = results[0]
        # results are dict like objects , rec_texts holds text line 
        if isinstance(res , dict):
            texts = res.get("rec_texts") or [] 
            return "\n".join(str(t) for t in texts if t)
        #Fallback 
        if hasattr(res,"rec_texts"):
            return "\n".join(str(t) for t in res.rec_texts if t)
        return "" 
    except AttributeError:
        #Graceful fallbacl to 2.x API if somehow older engine is used 
        result = ocr.ocr(path , cls=True) 
        if not result or result[0] is None:
            return ""
        return "\n".join(line[1][0] for line in results[0] if line and len(line)>=2)
    