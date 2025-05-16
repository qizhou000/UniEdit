from dataset.llm import UniEdit, Counterfact, ZSRE, BaseLLMEditData
from ...llms_for_edit import BaseLLMForEdit
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from utils import find_module
from .. import LLMBaseEditor
from ... import BaseConfig
from ... import nethook
from copy import deepcopy
from torch import nn
import numpy as np
import torch

@dataclass
class IKEConfig(BaseConfig):
    edit_model_name: str
    begin_layer_path: str
    lm_head_path: str
    max_demonstration_n: int
    max_demo_tokens: int

class IKE(LLMBaseEditor):
    def __init__(self, llm: BaseLLMForEdit, config:IKEConfig = IKEConfig, 
                 device='cuda:0', random_seed = 1):
        super().__init__(llm, device)
        self.cfg = config
        self.begin_layer = find_module(self.llm.model, self.cfg.begin_layer_path)
        self.lm_head = find_module(self.llm.model, self.cfg.lm_head_path)
        self.wrap_model_forward(self.llm, self.begin_layer, self.lm_head)
        self.register_editing_hook(self.begin_layer, self.lm_head)
        self.prepare_demo = True
        self.edit_fact_prefix = '<New Facts>:\n'
        self.edit_query_answer_prefix = '<Query Answer>:\n'
        demo_dataset = UniEdit('data/UniEdit/train', None, [], None, None, add_wiki_loc = False)
        self.edit_demonstration = self.get_edit_demonstration(self.edit_fact_prefix, 
            self.edit_query_answer_prefix, demo_dataset, random_seed)
        self.prepare_demo = False
        self.restore_to_original_model()

    def wrap_model_forward(self, model:BaseLLMForEdit, begin_layer, lm_head):
        if not hasattr(model, 'original_forward'):
            model.original_forward = model.model.forward
        def forward_ike(**kargs):
            if self.prepare_demo:
                return model.original_forward(**kargs)
            if 'past_key_values' in kargs and kargs['past_key_values'] != None:
                begin_layer.has_past_kv = True
                lm_head.has_past_kv = True
            else:
                begin_layer.has_past_kv = False
                lm_head.has_past_kv = False
                b, l = kargs['input_ids'].shape
                assert b == 1
                self.original_inpt_tok_len = l
                inpt_str = self.llm.tokenizer.decode(kargs['input_ids'][0], skip_special_tokens=True)
                kargs = self.llm.get_llm_input_ids(self.prefix_str + inpt_str)
            return model.original_forward(**kargs)
        model.model.forward = forward_ike

    def register_editing_hook(self, begin_layer:nn.Module, lm_head_layer:nn.Module):
        def lm_head_layer_forward_hook(module, args, output): # model output remove the prefix
            if self.prepare_demo:
                return 
            if not module.has_past_kv:
                assert output.shape[0] == 1 # batch size = 1
                output = output[:, -self.original_inpt_tok_len:]
            return output
        lm_head_layer._forward_hooks.clear()
        lm_head_layer.register_forward_hook(lm_head_layer_forward_hook)

    def get_edit_demonstration(self, edit_fact_prefix:str, edit_query_answer_prefix:str, 
                               dataset:BaseLLMEditData, random_seed = 1):
        rng = np.random.default_rng(random_seed)
        indxs = list(range(dataset.data_n()))
        rng.shuffle(indxs)
        edit_demonstration = ''
        now_demos = 0
        now_demo_toks = 0
        rs = []
        for i in indxs:
            d = dataset.get_data_by_ids([i])[0]
            if rs == []:
                rs = [1,2,3]
                rng.shuffle(rs)
            r = rs.pop()
            demo_facts = [{'prompt': d['requests'][0]['prompt'], 'target': d['requests'][0]['target_new']}]
            if r == 1:
                demo_prompt_ans = {'prompt': d['requests'][0]['prompt'], 
                                   'target': d['requests'][0]['target_new']}
            elif r == 2:
                gn = rng.choice(list(d['generality'].keys()))
                gd = d['generality'][gn][0]
                if 'related_hops' in gd:
                    for rh in gd['related_hops']:
                        demo_facts.append({'prompt': rh['prompt'], 'target': rh['target']})
                demo_prompt_ans = {'prompt': gd['prompt'], 'target': gd['target']}
            elif r == 3:
                ln = rng.choice(list(d['locality'].keys()))
                ld = d['locality'][ln][0]
                gen_str, all_ids, new_gen_ids = self.llm.generate(ld['prompt'], 
                    max_gen_toks = len(self.llm.tokenizer(ld['target'])['input_ids']), 
                    real_time_output = False)
                demo_prompt_ans = {'prompt': ld['prompt'], 
                                   'target': self.llm.tokenizer.decode(new_gen_ids)}
            new_demo = edit_fact_prefix
            for df in demo_facts:
                if df['prompt'][-1] != ' ' and df['target'][0] != ' ':
                    df['prompt'] += ' '
                new_demo += df['prompt'] + df['target'] + '\n'
            new_demo += edit_query_answer_prefix
            if demo_prompt_ans['prompt'][-1] != ' ' and demo_prompt_ans['target'][0] != ' ':
                demo_prompt_ans['prompt'] += ' '
            new_demo += demo_prompt_ans['prompt'] + demo_prompt_ans['target'] + '\n\n'
            edit_demonstration += new_demo
            now_demos += 1
            now_demo_toks += len(self.llm.tokenizer(new_demo)['input_ids'])
            if now_demos >= self.cfg.max_demonstration_n or now_demo_toks >= self.cfg.max_demo_tokens:
                print('Demostration count', now_demos)
                print('Demostration tokens:', now_demo_toks)
                break
        return edit_demonstration

    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'ike', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return False

    def restore_to_original_model(self):
        self.edit_prompts = []
        self.prefix_str = ''

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'prompt': str, 'target_new': str, ...} """
        p, t = request['prompt'], request['target_new']
        if p[-1] != ' ' and t[0] != ' ':
            p += ' '
        self.edit_prompts.append(p + t) 
        self.make_forward_prefix()
        
    def edit_batch(self, requests: List[Dict]):
        raise

    def save_current_edit_status(self):
        self.saved_edit_prompts_n = len(self.edit_prompts)

    def restore_to_saved_edit_status(self):
        self.edit_prompts = self.edit_prompts[:self.saved_edit_prompts_n]
        self.make_forward_prefix()

    def make_forward_prefix(self):
        self.prefix_str = self.edit_demonstration + self.edit_fact_prefix
        self.prefix_str += '\n'.join(self.edit_prompts)
        self.prefix_str += '\n' + self.edit_query_answer_prefix
