from ...llms_for_edit import BaseLLMForEdit
from utils import find_module, move_to_device
from .. import LLMBaseEditorWithTraining
from torch.nn.utils.rnn import pad_sequence
from dataset.llm import BaseLLMEditData
from typing import Dict, List, Tuple
from dataclasses import dataclass
from types import SimpleNamespace
from ... import BaseConfig
import os, torch, json, yaml
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm


@dataclass
class SERACConfig(BaseConfig):
    @dataclass
    class TrainConfig():
        lr: float
        relia_lambda: float
        gen_lambda: float
        loc_lambda: float
    edit_model_name: str
    counterfact_model_path: str
    counterfact_model_rep_dim: int
    classifier_path: str
    classifier_rep_dim: int
    llm_hidden_size: int
    train_config: TrainConfig
    llm_norm_path: str
    llm_voc_path: str
    train_classifier: bool
    
    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['train_config'] = self.TrainConfig(**data['train_config'])
        return self(**data)

class SERAC(LLMBaseEditorWithTraining):
    def __init__(self, llm: BaseLLMForEdit, config: SERACConfig, device:str = 'cuda:0'):
        super().__init__(llm, config, device)
        self.cfg = config
        # Initialize train modules
        from .modules import CounterfactModel, Classifier
        llm_norm = find_module(llm.model, config.llm_norm_path)
        voc_path = find_module(llm.model, config.llm_voc_path)
        def reps_to_word_predict(reps:torch.Tensor):
            return voc_path(llm_norm(reps))
        self.counterfact_model = CounterfactModel(config, reps_to_word_predict).to(device)
        self.classifier = Classifier(config, device).to(device)
        self.class_threshold = 20
        # hook llm `get_llm_outpt` function
        self.wrap_get_llm_outpt()
        self.restore_to_original_model()
        self.set_train(False)
    
    ############################################################################
    #                            Initialize                                    #
    ############################################################################
    def wrap_get_llm_outpt(self):
        def wrap(get_llm_outpt):
            def wrapped_get_llm_outpt(input_ids):
                if self.is_train or len(self.request_pool) == 0:
                    return get_llm_outpt(input_ids)
                assert len(input_ids['input_ids']) == 1 and len(input_ids['input_ids'].shape) == 2
                embd = self.get_embed_with_input_ids(input_ids)
                rep1 = self.classifier.get_sim_reps([self.llm.tokenizer.decode(
                    input_ids['input_ids'][0])])
                sim = self.classifier.get_sim(rep1, self.sim_reps_pool)
                v, i = torch.max(sim, dim=1) # [1], [1]
                if v[0] >= self.class_threshold:
                    logits = self.counterfact_model.forward_with_request_embd(
                        [self.request_embed_pool[i]], [embd['inputs_embeds']])[0]
                else:
                    logits = get_llm_outpt(input_ids).logits
                otpt = SimpleNamespace()
                otpt.logits = logits
                return otpt
            return wrapped_get_llm_outpt
        if not hasattr(self.llm, 'original_get_llm_outpt'):
            self.llm.original_get_llm_outpt = self.llm.get_llm_outpt
        self.llm.get_llm_outpt = wrap(self.llm.original_get_llm_outpt)

    ############################################################################
    #                          Utilized Functions                              #
    ############################################################################
    def get_embed_with_input_ids(self, input_ids:Dict):
        embeds = {'attention_mask': input_ids['attention_mask'],
            'inputs_embeds': self.llm.model.get_input_embeddings()(input_ids['input_ids'])}
        return embeds

    ############################################################################
    #                 Implementation Functions of Base Class                   #
    ############################################################################
    def name_of_editor_and_model(self):
        return 'serac', self.cfg.edit_model_name

    def if_can_batch_edit(self)->bool:
        return False

    def restore_to_original_model(self):
        self.sim_reps_pool = torch.zeros([0, self.cfg.classifier_rep_dim], device=self.device)
        self.request_embed_pool = []
        self.request_pool = []
        self.req_str_pool = []

    def edit_one_piece(self, request: Dict):
        """request = {'prompt': str, 'target_new': str, ...} """
        input_ids, _, _ = self.llm.prompts_targets_to_xym([request['prompt']], [request['target_new']])
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids['input_ids'])
        req_str = self.llm.tokenizer.decode(input_ids['input_ids'][0])
        reps = self.classifier.get_sim_reps(req_str)
        self.sim_reps_pool = torch.cat([self.sim_reps_pool, reps], 0)
        self.request_embed_pool.append(inputs_embeds)
        self.request_pool.append(request)
        self.req_str_pool.append(req_str)

    def edit_batch(self, requests: List[Dict]):
        raise

    def save_current_edit_status(self):
        self.current_edit_n = len(self.sim_reps_pool)

    def restore_to_saved_edit_status(self):
        self.sim_reps_pool = self.sim_reps_pool[:self.current_edit_n]
        self.request_embed_pool = self.request_embed_pool[:self.current_edit_n]
        self.request_pool = self.request_pool[:self.current_edit_n]
        self.req_str_pool = self.req_str_pool[:self.current_edit_n]

    ############################################################################
    #                       Training functions                                 #
    ############################################################################
    def get_modules_for_training(self)->Dict[str, nn.Module]:
        return {'counterfact_model': self.counterfact_model, 'classifier': self.classifier}
    
    def reinit_train_parameters(self):
        pass

    def preprocess_train_data(self, llm_edit_data:BaseLLMEditData)->List:
        return llm_edit_data
        
    def organize_batch_data(self, a_batch_of_training_data:List):
        llm = self.llm
        edit_xyms = []
        gen_xyms = {k: [] for k in a_batch_of_training_data[0]['generality'].keys()}
        loc_xyms = {k: [] for k in a_batch_of_training_data[0]['locality'].keys()}
        for d in a_batch_of_training_data:
            # edit/rel
            prompts = [d['requests'][0]['prompt']]
            targets = [d['requests'][0]['target_new']]
            edit_xyms.append((prompts, llm.prompts_targets_to_xym(prompts, targets)))
            # gen
            for k in gen_xyms.keys():
                prompts = [d['generality'][k][0]['prompt']]
                targets = [d['generality'][k][0]['target']]
                gen_xyms[k].append((prompts, llm.prompts_targets_to_xym(prompts, targets)))
            # loc
            for k in loc_xyms.keys():
                prompts = [d['locality'][k][0]['prompt']]
                targets = [d['locality'][k][0]['target']]
                input_ids, label_ids, label_masks = llm.prompts_targets_to_xym(prompts, targets)
                logits = self.llm.get_llm_outpt(input_ids).logits
                loc_xyms[k].append((prompts, (input_ids, logits, label_masks)))
        a_batch_of_training_data = move_to_device((edit_xyms, gen_xyms, loc_xyms), self.device)
        return a_batch_of_training_data
        
    def train_a_batch(self, a_batch_of_training_data:Tuple):
        edit_xyms, gen_xyms, loc_xyms = a_batch_of_training_data
        loss = 0  
        log_dict = {}
        batch_size = len(edit_xyms)
        # collect edit representations
        edit_input_embeds, edit_reps,  = [], []
        for _, (input_ids, _, _) in edit_xyms:
            input_embeds = self.get_embed_with_input_ids(input_ids)
            edit_input_embeds.append(input_embeds['inputs_embeds'])
            if self.cfg.train_classifier:
                edit_pt = self.llm.tokenizer.decode(input_ids['input_ids'][0])
                edit_reps.append(self.classifier.get_sim_reps(edit_pt))
        if self.cfg.train_classifier:
            edit_reps = torch.cat(edit_reps, 0) # [b, d]
        # reliability edit loss
        loss_rel_edit = 0
        rel_sim_reps = []
        for (rel_q, (input_ids, label_ids, label_masks)), eie in zip(edit_xyms, edit_input_embeds):
            embd = self.get_embed_with_input_ids(input_ids)
            logits = self.counterfact_model.forward_with_request_embd([eie], [embd['inputs_embeds']])[0]
            loss_rel_edit += label_loss(logits, label_ids, label_masks)
            if self.cfg.train_classifier:
                rel_sim_reps.append(self.classifier.get_sim_reps(rel_q))
        loss_rel_edit = loss_rel_edit / batch_size
        loss += loss_rel_edit 
        log_dict['Reliability edit loss'] = float(loss_rel_edit)
        if self.cfg.train_classifier:
            rel_sim_reps = torch.cat(rel_sim_reps) # [b, d]
            rel_sim = self.classifier.get_sim(rel_sim_reps, edit_reps) # [b, b]
            loss_rel_clas_rela = - torch.log(torch.diag(torch.softmax(rel_sim, 1)) + 1e-8).mean()
            loss_rel_clas_abs = (torch.exp(- torch.diag(rel_sim)) * 
                    (torch.diag(rel_sim) < self.class_threshold + 1)).mean()
            loss += loss_rel_clas_rela + loss_rel_clas_abs * 0.1
            log_dict['Reliability relative class loss'] = float(loss_rel_clas_rela)
            log_dict['Reliability absolute class loss'] = float(loss_rel_clas_abs)
        # generality edit loss
        for k in gen_xyms.keys():
            loss_gen_edit = 0
            gen_sim_reps = []
            for (gen_q, (input_ids, label_ids, label_masks)), eie in zip(gen_xyms[k], edit_input_embeds):
                embd = self.get_embed_with_input_ids(input_ids)
                logits = self.counterfact_model.forward_with_request_embd([eie], [embd['inputs_embeds']])[0]
                loss_gen_edit += label_loss(logits, label_ids, label_masks)
                if self.cfg.train_classifier:
                    gen_sim_reps.append(self.classifier.get_sim_reps(gen_q))
            loss_gen_edit = loss_gen_edit / batch_size
            loss += loss_gen_edit
            log_dict['Generality-%s edit loss'%k] = float(loss_gen_edit)
            if self.cfg.train_classifier:
                gen_sim_reps = torch.cat(gen_sim_reps) # [b, d]
                gen_sim = self.classifier.get_sim(gen_sim_reps, edit_reps)
                loss_gen_clas_rela = - torch.log(torch.diag(torch.softmax(gen_sim, 1)) + 1e-8).mean()
                loss_gen_clas_abs = (torch.exp(- torch.diag(gen_sim)) * 
                        (torch.diag(gen_sim) < self.class_threshold + 1)).mean()
                loss += loss_gen_clas_rela + loss_gen_clas_abs * 0.1
                log_dict['Generality-%s relative class loss'%k] = float(loss_gen_clas_rela)
                log_dict['Generality-%s absolute class loss'%k] = float(loss_gen_clas_abs)
        # locality edit loss
        for k in loc_xyms.keys():
            loss_loc_edit = 0
            loc_sim_reps = []
            for (loc_q, (input_ids, pre_logits, label_masks)), eie in zip(loc_xyms[k], edit_input_embeds):
                embd = self.get_embed_with_input_ids(input_ids)
                logits = self.counterfact_model.forward_with_request_embd([eie], [embd['inputs_embeds']])[0]
                loss_loc_edit += logit_KL_loss(logits, pre_logits, label_masks)
                if self.cfg.train_classifier:
                    loc_sim_reps.append(self.classifier.get_sim_reps(loc_q))
            loss_loc_edit = loss_loc_edit / batch_size
            loss += loss_loc_edit
            log_dict['Locality-%s edit loss'%k] = float(loss_loc_edit)
            if self.cfg.train_classifier:
                loc_sim_reps = torch.cat(loc_sim_reps) # [b, d]
                loc_sim = self.classifier.get_sim(loc_sim_reps, edit_reps)
                loc_sim = torch.cat([loc_sim, self.class_threshold + torch.zeros([loc_sim.shape[0], 1], device=self.device)], 1)
                loss_loc_clas = - torch.log(torch.softmax(loc_sim, 1) + 1e-8)[:, -1].mean()
                loss += loss_loc_clas
                log_dict['Locality-%s class loss'%k] = float(loss_loc_clas)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return float(loss), log_dict

    def get_a_new_optimizer(self)->torch.optim.Optimizer:
        return Adam([{'params': self.classifier.parameters(), 'lr': self.cfg.train_config.lr},
                {'params': self.counterfact_model.parameters(), 'lr': self.cfg.train_config.lr}])
    
    def set_train(self, is_train = False):
        self.is_train = is_train
        self.llm.model.train(False)
        self.llm.model.requires_grad_(False)
        self.classifier.requires_grad_(is_train)
        self.classifier.train(is_train)
        self.counterfact_model.requires_grad_(is_train)
        self.counterfact_model.train(is_train)

    def other_train_init_final(self):
        pass

def bin_cross_entropy(reps, y):
    # reps, y: [b, l]
    eps = 1e-8
    loss = - y * torch.log(reps + eps) - (1 - y) * torch.log(1 - reps + eps)
    return loss
 
def label_loss(logits, label_ids, masks, average = True):
    # logits: [batch_size, total_l, d], label_ids/masks: [batch_size, short_l]
    logits = logits[:, -label_ids.shape[1]:]
    log_pre_p = torch.log_softmax(logits, -1)
    log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, short_l]
    loss = -(log_pre_p * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss

def logit_KL_loss(logits1, logits2, masks, average = True):
    # logits1/logits2: [batch, total_l, voc_size], masks: [batch, short_l]
    logits1 = logits1[:, -masks.shape[1]:]
    logits2 = logits2[:, -masks.shape[1]:]
    log_p1 = torch.log_softmax(logits1, -1)
    log_p2 = torch.log_softmax(logits2, -1)
    p1 = torch.softmax(logits1, 2)
    kl_loss = (p1 * (log_p1 - log_p2)).sum(2) # [batch, max_len]
    loss = (kl_loss * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss
