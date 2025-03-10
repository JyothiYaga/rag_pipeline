# src/embedding.py
from sentence_transformers import SentenceTransformer
from src.logger import get_logger
from src.Abstract_class import PipelineComponent

logger = get_logger(__name__)

class Embedding(PipelineComponent):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, chunks):
        """Generate embeddings for a list of text chunks."""
        embeddings = self.model.encode(chunks, convert_to_tensor=True)
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        # Move tensor to CPU before it's converted to NumPy
        return embeddings.cpu()

    def process(self, chunks):
        """Full embedding process."""
        return self.generate_embeddings(chunks)
