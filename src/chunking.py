# src/chunking.py
import PyPDF2
from src.logger import get_logger
from src.Abstract_class import PipelineComponent

logger = get_logger(__name__)

class Chunking(PipelineComponent):
    def __init__(self, chunk_size=200):
        self.chunk_size = chunk_size

    def extract_text(self, pdf_path):
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + " "
            logger.info(f"Extracted text from {pdf_path}")
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
        return text

    def chunk_text(self, text):
        """Split text into chunks of defined size."""
        chunks = [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        logger.info(f"Chunked text into {len(chunks)} parts")
        return chunks

    def process(self, pdf_path):
        """Full processing of PDF into chunks."""
        text = self.extract_text(pdf_path)
        return self.chunk_text(text)
