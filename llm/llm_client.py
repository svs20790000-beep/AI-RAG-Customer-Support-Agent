import torch

class LLMClient:
    def __init__(self, engine_instance):
        self.engine = engine_instance

        # Following the pattern: self.path = self.cfg['paths']['knowledge_base']
        # We access 'cfg' via the engine instance
        self.gen_cfg = self.engine.cfg['generation']

        # Assigning individual values
        self.max_new_tokens = self.gen_cfg['max_new_tokens']
        self.temperature = self.gen_cfg['temperature']
        self.top_p = self.gen_cfg['top_p']
        
    def generate(self, prompt, max_new_tokens=150):
        """Executes model inference."""
        inputs = self.engine.gen_tok(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            #max_length=1024
            max_length=1536
        ).to(self.engine.DEVICE)
        
        with torch.no_grad():
            output = self.engine.gen_mod.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                #max_new_tokens=max_new_tokens,
                #temperature=0.7,
                do_sample=True
                #top_p=0.9
            )
        return self.engine.gen_tok.decode(output[0], skip_special_tokens=True)    