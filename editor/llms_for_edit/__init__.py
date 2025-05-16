from typing import Dict, List, Tuple, Optional, Union
from transformers import  AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from ..nethook import TraceDict, Trace
from copy import deepcopy
from torch import nn
import torch

def set_tokenizer_pad_id(tokenizer:AutoTokenizer, padding_side = 'right'):
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print('Set [pad_token] as [eos_token].')
    print('Padding side is set as "%s".'%padding_side)
    tokenizer.padding_side = padding_side

############################################################################
############################ LLM Wrap Class ################################
############################################################################
class BaseLLMForEdit():
    '''
    A wrap of LLM that first converts text into embedded representations of the 
    language model, and then achieves the subsequent inference.
    ''' 
    def __init__(self, model:Union[nn.Module, AutoModelForCausalLM], 
                 tokenizer:AutoTokenizer, device:str) -> None:
        super().__init__() 
        self.model = model
        self.tokenizer = tokenizer
        self.set_device(device)
        set_tokenizer_pad_id(tokenizer, padding_side = 'right')

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def prompts_targets_to_xym(self, prompts:List[str], targets:List[str]):
        '''
        Assume batch_size is `len(prompts)`, equals `len(targets)`.
        return (type, dtype, shape): 
            1. `input_ids`, output from `self.get_llm_input_embeds`:
                (torch.Tensor, float, [batch_size, l_total, d])
            2. `label_ids`, predict ids:
                (torch.Tensor, Long, [batch_size, l_short])
            3. `label_masks`, mask of predict ids for training: 
                (torch.Tensor, Long, [batch_size, l_short])
        The `l_total` is the length of all input tokens, the `l_short` is the length of tokens
            used to compute loss.
        '''
        tokenizer = self.tokenizer
        targets = deepcopy(targets)
        for i, t in enumerate(targets):
            targets[i] = t if t[0] == ' ' else ' ' + t
        input_strs, label_ids, label_masks = [], [], []
        min_prompt_tok_n = 999
        prompt_last_tok_pos = []
        for p, t in zip(prompts, targets):
            inpt_str = p + t
            input_strs.append(inpt_str)
            input_id = tokenizer(inpt_str, return_tensors="pt")['input_ids'][0]
            label_id = torch.roll(input_id, -1, 0) 
            mask = torch.zeros_like(label_id)
            prompt_tok = tokenizer(p)['input_ids']
            prompt_last_tok_pos.append(len(prompt_tok) - 1)
            if min_prompt_tok_n > len(prompt_tok):
                min_prompt_tok_n = len(prompt_tok)
            mask[len(prompt_tok)-1:-1] += 1
            label_ids.append(label_id)
            label_masks.append(mask)
        input_ids = self.get_llm_input_ids(input_strs).to(self.device)
        label_ids = pad_sequence(label_ids, True, tokenizer.pad_token_id).to(self.device)[:, min_prompt_tok_n - 1:]
        label_masks = pad_sequence(label_masks, True, 0).to(self.device)[:, min_prompt_tok_n - 1:]
        input_ids.prompt_last_tok_pos = prompt_last_tok_pos
        return input_ids, label_ids, label_masks

    # def get_target_prob_distribution(self, prompt:str, target:str):
    #     # Inference with the LLM given prompt and target, and get the prediction prob distribution of the target
    #     input_ids, label_ids, label_masks = self.prompts_targets_to_xym([prompt], [target])
    #     label_masks = label_masks[0]
    #     logit = self.get_llm_outpt(input_ids).logits[0, -len(label_masks):][label_masks.to(bool)]
    #     prob = torch.softmax(logit, 1)
    #     return prob # [answer_length, vocabulary_length]

    def generate(self, prompt:str, use_past_key_values:bool = True, real_time_output = True, 
                 max_gen_toks = 10)->Tuple[str, torch.Tensor]: # generate tokens until meet eos
        with torch.no_grad():
            inpt = self.get_llm_input_ids([prompt])
            all_ids = inpt['input_ids'][0]
            new_gen_ids = []
            last_tok_id = all_ids[-1]
            if real_time_output:
                print(prompt, end='')
            while last_tok_id != self.tokenizer.eos_token_id and len(new_gen_ids) <= max_gen_toks:
                otpt = self.get_llm_outpt(inpt) # [1, 1, V]
                last_tok_id = torch.argmax(torch.softmax(otpt.logits[:1, -1:], -1), -1) # [1, 1]
                new_gen_ids.append(int(last_tok_id))
                all_ids = torch.cat([all_ids, last_tok_id[0]])
                if real_time_output:
                    print(self.tokenizer.decode(last_tok_id[0, 0]), end='')
                if use_past_key_values:
                    inpt = {
                        'past_key_values': otpt.past_key_values,
                        'input_ids': last_tok_id
                    }
                else:
                    inpt = {
                        'attention_mask': torch.cat([inpt['attention_mask'], torch.ones([1, 1], 
                            dtype=torch.long, device=self.device)], 1),
                        'input_ids': torch.cat([inpt['input_ids'], last_tok_id], 1),
                    }
        gen_str = self.tokenizer.decode(all_ids)
        return gen_str, all_ids, new_gen_ids

    def get_llm_input_ids(self, texts:List[str]):
        return self.tokenizer(texts, padding = True, return_tensors="pt").to(self.device)

    def get_llm_outpt(self, input_ids:Dict):
        # input_ids: {'input_ids': torch.Tensor, 'attention_masks': torch.Tensor}
        return self.model(**input_ids)
        
    def label_loss(self, logits, label_ids, label_masks, average = True):
        # logits: [batch_size, total_l, d], label_ids/label_masks: [batch_size, short_l]
        logits = logits[:, -label_ids.shape[1]:]
        log_pre_p = torch.log_softmax(logits, -1)
        log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, short_l]
        loss = -(log_pre_p * label_masks).sum()
        if average:
            loss = loss / label_masks.sum() 
        return loss
        
    def hinge_label_loss(self, logits:torch.Tensor, label_ids:torch.Tensor, 
                masks:torch.Tensor, average = True, eps = 1e-8, hinge_scale = 1.1):
        # input_ids: [batch, max_len]; label_ids/masks: [batch, label_len]
        assert label_ids.shape[1] == masks.shape[1]
        pre_p = torch.softmax(logits[:, -label_ids.shape[1]:], 2) # [batch, label_len, voc_size]
        second_pre_p = torch.topk(pre_p, 2, -1).values[:, :, 1] # [batch, label_len]
        pre_p = pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, label_len]
        masks = masks * (pre_p < second_pre_p * hinge_scale)
        pre_p = torch.masked_select(pre_p, masks.to(bool))
        loss = - torch.log(pre_p + eps).sum() # [batch, max_len] 
        sm = masks.sum() 
        if average:
            if sm != 0:
                loss = loss / sm
            return loss
        else:
            return loss, sm

    def logit_KL_loss(self, logits1, logits2, label_masks, average = True):
        # logits1/logits2: [batch, total_l, voc_size], label_masks: [batch, short_l]
        logits1 = logits1[:, -label_masks.shape[1]:]
        logits2 = logits2[:, -label_masks.shape[1]:]
        log_p1 = torch.log_softmax(logits1, -1)
        log_p2 = torch.log_softmax(logits2, -1)
        p1 = torch.softmax(logits1, 2)
        kl_loss = (p1 * (log_p1 - log_p2)).sum(2) # [batch, max_len]
        loss = (kl_loss * label_masks).sum()
        if average:
            loss = loss / label_masks.sum() 
        return loss

    def get_mid_module_inpt(self, input_toks, mid_module_path, get_first_if_tuple = True):
        with torch.no_grad(), Trace(self.model, mid_module_path, retain_input = True, 
                with_kwargs = False, stop = True, clone = False) as t:
            self.get_llm_outpt(input_toks)
        if isinstance(t.input, torch.Tensor):
            return t.input
        elif isinstance(t.input, (list, tuple)):
            if get_first_if_tuple:
                return t.input[0] # args[0] is the middle layer representation
            return t.input # args[0] is the middle layer representation
        raise "Unknown Type!"

    def get_mid_module_outpt(self, input_toks, mid_module_path,
                             get_first_if_tuple = True):
        with torch.no_grad(), Trace(self.model, mid_module_path, retain_output = True, 
                with_kwargs = False, stop = True, clone = False) as t:
            self.get_llm_outpt(input_toks)
        if isinstance(t.output, torch.Tensor):
            return t.output # output[0] is middle layer representation 
        elif isinstance(t.output, (list, tuple)):
            if get_first_if_tuple:
                return t.output[0] # output[0] is middle layer representation 
            return t.output # output[0] is middle layer representation 
        raise "Unknown Type!"
    
    def forward_from_mid_layer(self, input_toks, mid_layer_inpt:torch.Tensor, 
                             llm_layer_tmp:str, mid_inpt_layer_i:int):
        ''' Use `mid_layer_inpt` to inference from LLM's middle layer for speed up inference.
        Note! Only support basic layer, and do not support other modules like Att and FFN.
        The shape of hidden representation of `input_toks` must be same with 
        `mid_layer_inpt`. 
        `llm_layer_tmp`: the layer name template. 
        `mid_inpt_layer_i`: the index of the middle layer to begin inference.
        '''
        def skip_layer(*args, **kargs):
            return [None]*2
        def mid_inpt_embeds(inpt, layer):
            args, kargs = td[llm_layer_tmp.format(0)].input
            args = (mid_layer_inpt,)
            return args, kargs
        skip_layers = [llm_layer_tmp.format(i) for i in range(mid_inpt_layer_i)]
        inpt_layer = [llm_layer_tmp.format(mid_inpt_layer_i)]
        with TraceDict(self.model, [llm_layer_tmp.format(0)], retain_input=True
            ) as td, TraceDict(self.model, skip_layers, layer_func_replace=skip_layer
            ), TraceDict(self.model, inpt_layer, edit_input=mid_inpt_embeds): 
            outpt = self.get_llm_outpt(input_toks)
        return outpt

    def find_closest_tokens(self, embeddings:torch.Tensor, embedding_matrix:torch.Tensor, top_k = 1):
        # embeddings: [b, l, d]; embedding_matrix: [v, d]
        # Normalization
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        embedding_matrix_norm = embedding_matrix / embedding_matrix.norm(dim=1, keepdim=True)
        # Similarity
        cosine_sim = torch.matmul(embeddings_norm, embedding_matrix_norm.T)
        top_k_indices = torch.topk(cosine_sim, top_k, dim=-1)
        toks, sim = top_k_indices.indices, top_k_indices.values
        return toks, sim
