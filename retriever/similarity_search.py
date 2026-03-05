import faiss
import numpy as np

class SimilaritySearch:
    def __init__(self, index):
        self.index = index

    def perform_search(self, query_emb, k=3):
        """Execute the FAISS L2 distance search."""
        # Ensure query is float32 and 2D
        query_emb = query_emb.astype('float32')
        if len(query_emb.shape) == 1:
            query_emb = np.expand_dims(query_emb, axis=0)
            
        distances, indices = self.index.search(query_emb, k)
        return distances, indices