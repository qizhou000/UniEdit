
from .. import BaseLLMForEdit

class LLAMA3ForEdit(BaseLLMForEdit):
    def __init__(self, model_path:str = 'models/llama-3-8b', device = 'cuda') -> None:
        from transformers import LlamaForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path)
        super().__init__(model, tokenizer, device)
