from .prompt_templates import PromptTemplates
from .llm_client import LLMClient

class ResponseGenerator:
    def __init__(self, engine_instance):
        self.client = LLMClient(engine_instance)
        self.templates = PromptTemplates()

    def generate_final_response(self, query, context_chunks):
        """Combines context and query into a final answer."""
        context_str = "\n".join(context_chunks)
        prompt = self.templates.get_qa_prompt(context_str, query)
        
        return self.client.generate(prompt)