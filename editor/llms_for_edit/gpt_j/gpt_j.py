
from .. import BaseLLMForEdit

class GPTJForEdit(BaseLLMForEdit):
    def __init__(self, model_path:str = 'models/gpt-j-6b', device = 'cuda') -> None:
        from transformers import AutoTokenizer, GPTJForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GPTJForCausalLM.from_pretrained(model_path)
        super().__init__(model, tokenizer, device)
