import numpy as np

class RAGEvaluator:
    @staticmethod
    def calculate_hit_rate(retrieved_chunks, keywords):
        """Checks if the correct information was even found."""
        for chunk in retrieved_chunks:
            if any(key.lower() in chunk.lower() for key in keywords):
                return 1
        return 0

    @staticmethod
    def simple_exact_match(predicted, expected):
        """Basic check for overlap (useful for T5)."""
        return 1 if expected.lower() in predicted.lower() else 0
