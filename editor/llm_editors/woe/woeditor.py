from ... import BaseConfig
from .. import LLMBaseEditor
from ...llms_for_edit import BaseLLMForEdit
from typing import Dict, List, Tuple 
from dataclasses import dataclass
 
@dataclass
class WOEConfig(BaseConfig):
    edit_model_name: str

class WOE(LLMBaseEditor):
    def __init__(self, model: BaseLLMForEdit, config: WOEConfig, device = 'cuda'):
        super().__init__(model, device)
        self.cfg = config

    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'woe', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return True
 
    def restore_to_original_model(self):
        pass

    def edit_batch(self, requests: List[Dict]):
        '''requests = [
            {'prompt':str, 'subject':str, 'target_new':str}
            {'prompt':str, 'subject':str, 'target_new':str}, ...
        ]'''
        pass

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'prompt':str, 'subject':str, 'target_new':str}"""
        pass

    def save_current_edit_status(self):
        pass

    def restore_to_saved_edit_status(self):
        pass

