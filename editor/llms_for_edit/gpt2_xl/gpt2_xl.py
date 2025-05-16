
from .. import BaseLLMForEdit

class GPT2XLForEdit(BaseLLMForEdit):
    def __init__(self, model_path:str = 'models/gpt2-xl', device = 'cuda') -> None:
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        super().__init__(model, tokenizer, device)
