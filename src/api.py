from fastapi import FastAPI, UploadFile, File, HTTPException
from src.chunking import Chunking
from src.embedding import Embedding
from src.vectorization import Vectorization
from src.retrieval import Retrieval
import numpy as np
import os

app = FastAPI()

# Initialize components
chunker = Chunking()
embedder = Embedding()
vector_db = Vectorization(dim=384)
retriever = Retrieval(dim=384)

# Connect the retriever to the vector database
retriever.set_index(vector_db.index)

# Add storage for chunks to allow returning them with results
text_chunks = []

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global text_chunks
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    pdf_path = f"uploads/{file.filename}"
    
    # Save file
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Process PDF
        chunks = chunker.process(pdf_path)
        text_chunks = chunks  # Store chunks for later retrieval
        
        embeddings = embedder.process(chunks)
        # Make sure embeddings are on CPU before converting to NumPy
        if hasattr(embeddings, 'is_cuda') or hasattr(embeddings, 'device'):
            embeddings = embeddings.cpu()
            
        vector_db.process(np.array(embeddings, dtype=np.float32))
        
        # Reconnect retriever with the updated index
        retriever.set_index(vector_db.index)
        
        return {"message": "PDF processed successfully", "chunks_count": len(chunks)}
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}\n{error_detail}")

@app.get("/query/")
async def query_rag(question: str):
    try:
        if not text_chunks:
            raise HTTPException(status_code=400, detail="No documents have been processed yet")
            
        indices = retriever.process(question)
        
        # Return both indices and the corresponding text chunks
        results = [
            {"index": int(idx), "text": text_chunks[idx] if idx < len(text_chunks) else "Index out of range"} 
            for idx in indices
        ]
        
        return {"results": results}
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error querying: {str(e)}\n{error_detail}")