import numpy as np 
from sentence_transformers import SentenceTransformer 



from config.settings import EMBEDDING_MODEL 


class EmbeddingService:
    _instance: "EmbeddingService | None" = None  #for singleton pattern

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
    
    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None: #only make instance if instance is None i.e first instance

            cls._instance = cls() 
        return cls._instance 


    def encode(self, texts: list[str], batch_size: int=32) -> np.ndarray:
        vecs = self._model.encode(
            texts , batch_size= batch_size , normalize_embeddings=True , show_progress_bar=False 
        )

        return vecs.astype(np.float32)

    def encode_single(self, text:str) -> np.ndarray:
        return self.encode([text])[0]
