import cv2
import shutil 
import os 
import imagehash 
from PIL import Image
import numpy as np 
import xml.etree.ElementTree as ET
from typing import List
from models.state import ContractState
from utils.file_utils import pdf_to_images, make_temp_dir

def _deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate rotation from text block contours and correct it."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = [c for c in contours if cv2.contourArea(c) > 200]
    if len(large) < 5:
        return gray
    all_pts = np.vstack([c.reshape(-1, 2) for c in large])
    rect = cv2.minAreaRect(all_pts)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 0.5:
        return gray  # already straight
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def enhance_image(path: str) -> None:
    """In-place: denoise → CLAHE contrast → deskew."""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Stage 1: Remove salt-and-pepper noise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    # Stage 2: Adaptive contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Stage 3: Correct rotation using text block contours
    gray = _deskew(gray)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path, out)


#page deduplication (perpetual hash)
def _dedup_pages(paths: list[str], threshold:int=6):
    hashes = [] 
    kept = [] 
    for i , j in enumerate(paths):
        h = imagehash.phash(Image.open(p))
        # keep the page if its hash differs by more than threshold bits from all kept pages 
        if all((h-prev)>threshold for prev in hashes):
            kept.append(i)
            hashes.append(h)
        return kept 


# XM parswer 

def _xml_to_text(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _recurse(elem: ET.Element, indent: int = 0) -> List[str]:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        text = (elem.text or "").strip()
        lines: List[str] = []
        prefix = "  " * indent
        if text:
            lines.append(f"{prefix}{tag}: {text}")
        else:
            lines.append(f"{prefix}{tag}:")
        for attr_k, attr_v in elem.attrib.items():
            lines.append(f"{prefix}  @{attr_k}: {attr_v}")
        for child in elem:
            lines.extend(_recurse(child, indent + 1))
        tail = (elem.tail or "").strip()
        if tail:
            lines.append(f"{prefix}{tail}")
        return lines

    return "\n".join(_recurse(root))


# Node 

def preprocess_node(state: ContractState) -> dict:
    log = list(state.get("processing_log", []))
    file_type = state["file_type"]

    # XML: no image pipeline needed
    if file_type == "xml":
        text = _xml_to_text(state["file_path"])
        log.append(f"XML parsed → {len(text)} chars")
        return {
            **state,
            "page_image_paths": [],
            "deduplicated_page_indices": [],
            "raw_text_by_page": {0: text},
            "full_text": text,
            "processing_log": log,
            "current_step": "ocr_extraction",
        }

    # PDF or image → rasterise
    tmp = make_temp_dir()
    if file_type == "pdf":
        raw_paths = pdf_to_images(state["file_path"], tmp)
    else:
        dest = os.path.join(tmp, "page_0000.png")
        shutil.copy(state["file_path"], dest)
        raw_paths = [dest]

    # Enhance
    for p in raw_paths:
        enhance_image(p)

    # Deduplicate
    kept_indices = _dedup_pages(raw_paths)
    kept_paths = [raw_paths[i] for i in kept_indices]

    log.append(
        f"Pre-processing: {len(raw_paths)} pages, "
        f"{len(kept_paths)} kept after dedup"
    )
    return {
        **state,
        "page_image_paths": kept_paths,
        "deduplicated_page_indices": kept_indices,
        "processing_log": log,
        "current_step": "ocr_extraction",
    }