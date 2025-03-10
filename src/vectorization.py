# src/vectorization.py
import faiss
import numpy as np
from src.logger import get_logger
from src.Abstract_class import PipelineComponent

logger = get_logger(__name__)

class Vectorization(PipelineComponent):
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def store_embeddings(self, embeddings):
        """Stores embeddings in FAISS vector DB."""
        self.index.add(embeddings)
        logger.info(f"Stored {embeddings.shape[0]} embeddings in FAISS")
    
    def process(self, embeddings):
        """Full vectorization process."""
        self.store_embeddings(embeddings)
