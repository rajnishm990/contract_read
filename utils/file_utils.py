import os 
import shutil 
from pathlib import Path 
import tempfile
from typing import Dict , List 

import fitz 

from config.settings import TEMP_DIR 

def get_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower() 

    if ext == ".pdf":
        return "pdf"
    elif ext in {",jpg", ".png", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        return "image"
    elif ext == ".xml":
        return "xml"
    raise ValueError(f"unsupported file type: {ext}. Supported: PDF , JPEG , JPG")


def pdf_to_image(pdf_path:str , output_dir : str , dpi:int=200)->List[str]:
    doc = fitz.open(pdf_path)
    paths: List[str] = [] 
    scale = dpi /72.0 
    mat = fitz.Matrix(scale , scale)

    for page_num , page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat , alpha=False)
        out = os.path.join(output_dir , f"page_{page_num:04d}.png")
        pix.save(out)
        paths.append(out) 
    doc.close() 
    return paths 

def extract_native_pdf_text(pdf_path: str)->Dict[int , str]:
    doc = fitz.open(pdf_path)
    result :Dict[int , str] = {}
    for page_num , page in enumerate(doc):
        result[page_num] = page.get_text()
    doc.close() 

    return result

def make_temp_dir() ->str:
    return tempfile.mkdtemp(dir= str(TEMP_DIR)) 

def clean_up_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
