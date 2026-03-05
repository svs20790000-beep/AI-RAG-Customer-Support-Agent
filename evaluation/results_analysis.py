import json
import pandas as pd
import logging
from .evaluation_metrics import RAGEvaluator

logger = logging.getLogger(__name__)

class EvaluationRunner:
    def __init__(self, engine_instance):
        self.engine = engine_instance
        self.evaluator = RAGEvaluator()

        # Following the pattern: self.path = self.cfg['paths']['knowledge_base']
        # We access 'cfg' via the engine instance
        self.gen_cfg = self.engine.cfg['retrieval']

        # Assigning individual values
        self.k_neighbors = self.gen_cfg['k_neighbors']
        

    def run_full_test(self, test_file_path="./evaluation/test_queries.json"):
        with open(test_file_path, 'r') as f:
            tests = json.load(f)
        
        results = []
        for test in tests:
            query = test['query']
            
            # Test Retrieval
            #retrieved = self.engine.retrieve(query, k=3)
            retrieved = self.engine.retrieve(query, k=self.k_neighbors)
            hit = self.evaluator.calculate_hit_rate(retrieved, test['context_keywords'])
            
            # Test Generation
            prediction = self.engine.generate_answer(query)
            accuracy = self.evaluator.calculate_hit_rate([prediction], [test['expected_answer']])
            
            results.append({
                "Query": query,
                "Retrieval_Hit": hit,
                "Correct_Answer": accuracy,
                "Model_Response": prediction
            })
            
        df = pd.DataFrame(results)
        df.to_csv("./evaluation/latest_results.csv", index=False)
        logger.info(f"Test Complete. Avg Accuracy: {df['Correct_Answer'].mean()}")
        return df
