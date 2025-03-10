import faiss
import numpy as np
from src.logger import get_logger
from src.Abstract_class import PipelineComponent
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

class Retrieval(PipelineComponent):
    def __init__(self, dim, model_name="all-MiniLM-L6-v2"):
        self.dim = dim
        self.model = SentenceTransformer(model_name)
        # Don't create a new index here as it should be shared
        self.index = None

    def set_index(self, index):
        """Set the FAISS index from vectorization component"""
        self.index = index
        logger.info("FAISS index has been set in Retrieval component")

    def search(self, query, top_k=3):
        """Finds top-k relevant chunks."""
        if self.index is None:
            logger.error("FAISS index not set. Call set_index() first.")
            raise ValueError("Index not initialized. Please process documents first.")
            
        query_embedding = self.model.encode(query)
        # Convert to numpy array and ensure it's on CPU
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        D, I = self.index.search(query_embedding, top_k)
        logger.info(f"Retrieved top {top_k} matches")
        return I[0]  # Return just the indices array without the extra dimension

    def process(self, query):
        """Full retrieval process."""
        return self.search(query)