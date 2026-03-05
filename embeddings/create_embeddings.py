import os
import faiss
import numpy as np
import torch
import json
import logging

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self, engine_instance):
        self.engine = engine_instance
        self.vector_store_path = "./embeddings/vector_store/chromadb" # Folder for persistence
        os.makedirs(self.vector_store_path, exist_ok=True)

    def create_and_save(self, texts):
        """Vectorizes texts and saves the index to disk."""
        logger.info(f"Starting vectorization of {len(texts)} chunks...")
        
        embeddings = []
        # Batching for efficiency
        batch_size = 8 
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.engine.ctx_tok(batch, return_tensors='pt', truncation=True, 
                                       padding=True).to(self.engine.DEVICE)
            
            with torch.no_grad():
                emb = self.engine.ctx_mod(**inputs).pooler_output.cpu().numpy()
            embeddings.append(emb)

        embeddings = np.vstack(embeddings)
        dimension = embeddings.shape[1]
        
        # Build FAISS Index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # PERSISTENCE: Save the index for later use
        index_file = os.path.join(self.vector_store_path, "index.faiss")
        faiss.write_index(index, index_file)
        
        logger.info(f"Vector store saved at {index_file}")
        return index

    def load_index(self):
        """Loads a pre-computed index from disk."""
        index_file = os.path.join(self.vector_store_path, "index.faiss")
        if os.path.exists(index_file):
            logger.info("Loading existing vector store...")
            return faiss.read_index(index_file)
        return None