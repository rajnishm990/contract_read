from typing import Dict  

from paddleocr import PaddleOCR 

from models.state import ContractState 
from services.llm import LLMService 
from utils.file_utils import extract_native_pd_text