class PromptTemplates:
    @staticmethod
    def get_qa_prompt(context, query):
        """Standardized prompt for FLAN-T5."""
        return (
            "Answer the following question using only the provided context. "
            "If the answer is not in the context, say 'I do not have enough information.'\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )