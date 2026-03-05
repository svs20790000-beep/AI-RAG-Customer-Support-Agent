import torch
import logging
from .similarity_search import SimilaritySearch

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, engine_instance):
        self.engine = engine_instance
        self.search_engine = SimilaritySearch(self.engine.faiss_index)

    def get_relevant_chunks(self, query, k=3):
        """Encodes query and returns the matching text chunks."""
        # 1. Encode the question using DPR Question Encoder
        inputs = self.engine.q_tok(
            query, 
            return_tensors='pt', 
            truncation=True, 
            padding=True
        ).to(self.engine.DEVICE)
        
        with torch.no_grad():
            query_emb = self.engine.q_mod(**inputs).pooler_output.cpu().numpy()

        # 2. Perform Similarity Search
        distances, indices = self.search_engine.perform_search(query_emb, k)
        
        # 3. Map indices back to text chunks
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.engine.knowledge_texts):
                results.append(self.engine.knowledge_texts[idx])
        
        logger.info(f"Retrieved {len(results)} chunks for query: '{query}'")
        return results