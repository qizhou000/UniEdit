#%% Load all final data
from uniedit_construction.tools.final_edit_data_generate import FinalEditDataGenerator
from numpy.random._generator import Generator as RNG
from typing import Dict, List, Tuple
from datetime import datetime
from copy import deepcopy
import argparse, json, os
from tqdm import tqdm
from time import time
import numpy as np


class FinalDataValidator():
    def __init__(self, subject:str, validate_threthold:float = 0.8, verbose = True):
        self.fedg = FinalEditDataGenerator()
        self.subject = subject
        self.val_threthold = validate_threthold
        self.verbose = verbose
        # Load final data for this subject
        final_data_dir = os.path.join('data/wikidata/s13_final_data', subject)
        self.final_data = self.fedg.load_and_merge_final_data(final_data_dir)
        print(subject, len(self.final_data))
        # Load invalid parts
        original_invalid_history_path = os.path.join('data/wikidata/s13_final_data/0_invalid_parts', '%s.json'%subject)
        self.original_invalid_history = {}
        if os.path.exists(original_invalid_history_path):
            with open(original_invalid_history_path, 'r') as f:
                self.original_invalid_history = json.load(f)
    
    def validate_target_with_right_answers(self, tgt:str, right_answers:List[str], 
                                           error_answers:List[str] = []):
        tgt_rm_space = ''.join(tgt.split(' '))
        # match error answers
        max_error_match_score = 0
        matched_error_ans = ''
        for ea in error_answers:
            ea_rm_space = ''.join(ea.split(' '))
            match_str = self.fedg.get_maxmum_common_substr(tgt_rm_space, ea_rm_space)
            match_score = len(match_str) / len(ea_rm_space)
            if match_score > max_error_match_score:
                max_error_match_score = match_score
                matched_error_ans = ea
        # match right answers: must larger than threshold and larger than error match score
        max_right_match_score = 0
        matched_right_ans = ''
        for ra in right_answers:
            ra_rm_space = ''.join(ra.split(' '))
            match_str = self.fedg.get_maxmum_common_substr(tgt_rm_space, ra_rm_space)
            match_score = len(match_str) / len(ra_rm_space)
            if match_score > max_right_match_score:
                max_right_match_score = match_score
                matched_right_ans = ra
        is_matched = max_right_match_score > self.val_threthold and max_right_match_score > max_error_match_score
        matched_results = {'right_match': [matched_right_ans, max_right_match_score], 
                           'error_match': [matched_error_ans, max_error_match_score]}
        return is_matched, matched_results
    
    def clear_target_and_prompt(self, d:dict, data_info:str):
        if self.verbose:
            print('Clear %s prompt and target.'%data_info)
        d['prompt'] = None
        d['target'] = None

    def get_and_mod_error_edit_data(self, edit_data, idx): 
        head_ent = edit_data['head_entity']
        tail_ent = edit_data['tail_entity']
        if tail_ent['datatype'] == 'wikibase-item':
            right_answers = [tail_ent['value']['label'], *tail_ent['value']['aliases']]
        else:
            right_answers = [self.fedg.get_triple_entity_label(tail_ent, 'tail_entity')]
        error_answers = [head_ent['label'], *head_ent['aliases']]
        matched, matched_results = self.validate_target_with_right_answers(
            edit_data['target'], right_answers, error_answers)
        if matched:
            return {}
        # AI generated target not match labels
        data_info = '%s-%s-edit'%(self.subject, idx)
        old_target = edit_data['target']
        self.clear_target_and_prompt(edit_data, data_info)
        return {'edit': {"right_answers": right_answers, 'target': old_target, 
                        'match_score': matched_results}} 

    def get_error_one_hop(self, one_hop): 
        # Judge if the ont hop has valid target given triple
        data_sign = one_hop['path'][0 if one_hop['reverse_in_multi_hop'] else 2].split('_')[1]
        right_ent_type = 'head_entity' if one_hop['reverse_in_multi_hop'] else 'tail_entity'
        error_ent_type = 'tail_entity' if one_hop['reverse_in_multi_hop'] else 'head_entity'
        # get right answers
        if data_sign == 'e' and (right_ent_type == 'head_entity' or one_hop[right_ent_type]['datatype'] == 'wikibase-item'):
            ent = one_hop[right_ent_type] if right_ent_type == 'head_entity' else one_hop[right_ent_type]['value']
            right_answers = [ent['label'], *ent['aliases']]
        else:
            right_answers = [self.fedg.get_triple_entity_label(one_hop[right_ent_type], right_ent_type)]
        # get error answers
        if data_sign == 'e' and (error_ent_type == 'head_entity' or one_hop[error_ent_type]['datatype'] == 'wikibase-item'):
            ent = one_hop[error_ent_type] if error_ent_type == 'head_entity' else one_hop[error_ent_type]['value']
            error_answers = [ent['label'], *ent['aliases']]
        else:
            error_answers = [self.fedg.get_triple_entity_label(one_hop[error_ent_type], error_ent_type)]
        # match 
        onehop_tgt = one_hop['reversed']['target'] if one_hop['reverse_in_multi_hop'] else one_hop['target']
        matched, matched_results = self.validate_target_with_right_answers(
            onehop_tgt, right_answers, error_answers)
        if matched:
            return 'right', {"right_answers": right_answers, 'error_answers': error_answers}
        return 'error', {"right_answers": right_answers, 'error_answers': error_answers, 
            'target': onehop_tgt, 'reverse': one_hop['reverse_in_multi_hop'], 'match_score': matched_results}

    def get_and_mod_error_single_path(self, single_path, idx, gen_loc, gen_loc_i, single_double): 
        invalid = {}
        for path_i, one_hop in enumerate(single_path['one_hops']):
            state, match_res = self.get_error_one_hop(one_hop)
            if state == 'error': 
                invalid[path_i] = match_res
        last_hop_right_answers = match_res['right_answers']
        last_hop_error_answers = match_res['error_answers']
        matched, matched_results = self.validate_target_with_right_answers(
            single_path['target'], last_hop_right_answers, last_hop_error_answers)
        data_info = '%s-%s-%s-%s-%s'%(self.subject, idx, gen_loc, gen_loc_i, single_double)
        if len(invalid) > 0:
            invalid['multi_hop'] = {'right_answers': last_hop_right_answers, 
                'error_answers': last_hop_error_answers, 
                'target': single_path['target'], 'match_score': matched_results}
            self.clear_target_and_prompt(single_path, data_info)
        else: # len(invalid) == 0:
            if not matched: 
                invalid['multi_hop'] = {'right_answers': last_hop_right_answers, 
                    'error_answers': last_hop_error_answers,
                    'target': single_path['target'], 'match_score': matched_results}
                self.clear_target_and_prompt(single_path, data_info)
        return invalid

    def update_and_mod_invalid_parts(self, idx):
        invalid_parts = {}
        def update_invalid_history(k, new_invalid):
            if k in self.original_invalid_history.keys():
                invalid_parts[k] = deepcopy(self.original_invalid_history[k])
                invalid_parts[k]['invalid_count'] += 1
                invalid_parts[k]['history'].append(new_invalid)
            else:
                invalid_parts[k] = {'invalid_count': 1, 'history': [new_invalid]}
        d = self.final_data[idx]
        # for edit
        if self.fedg.has_valid_pt(d, ['edit']):
            ied = self.get_and_mod_error_edit_data(d['edit'], idx)
            if len(ied) != 0: 
                update_invalid_history('%s-%s-edit'%(self.subject, idx), ied)
        else:
            update_invalid_history('%s-%s-edit'%(self.subject, idx), 'empty')
        # for generality and locality
        for gl in ['generality', 'locality']:
            for gli, gld in d[gl].items(): 
                info = '%s-%s-%s-%s'%(self.subject, idx, gl, gli)
                if 'path_type' not in gld.keys():
                    update_invalid_history(info, 'empty')
                    continue
                info = '%s-%s'%(info, gld['path_type'])
                if gld['path_type'] == 'single':
                    if not self.fedg.has_valid_pt(gld, []):
                        update_invalid_history(info, 'empty')
                    else:
                        esp = self.get_and_mod_error_single_path(gld, idx, gl, gli, 'single')
                        if len(esp) != 0:
                            update_invalid_history(info, esp)
                else: # double
                    if not self.fedg.has_valid_pt(gld, []):
                        update_invalid_history(info, 'empty')
                    for dpi in [1, 2]: # double_path_i
                        a_path = gld['single_path%s'%dpi]
                        path_info = '%s-path_%s'%(info, dpi)
                        if not self.fedg.has_valid_pt(a_path, []):
                            update_invalid_history(path_info, 'empty')
                            self.clear_target_and_prompt(gld, info) # Clear double path target and prompt
                        else:
                            esp = self.get_and_mod_error_single_path(a_path, idx, gl, gli, 'double_path_%s'%dpi)
                            if len(esp) != 0:
                                update_invalid_history(path_info, esp)
                                self.clear_target_and_prompt(gld, info) # Clear double path target and prompt
        return invalid_parts
    
# all_subjects = ['biology', 'mathematics', 'chemistry', 'physics', 
#     'geoscience', 'astronomy', 'sociology', 'jurisprudence', 
#     'political science', 'economics', 'psychology', 'pedagogy', 
#     'civil engineering', 'mechanical engineering', 'medicine', 
#     'computer science', 'agronomy', 'literature', 'history', 
#     'philosophy', 'art', 'material science', 'environmental science', 
#     'sports science', 'data science']

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--subject', type=str, required=True)
    args = parser.parse_args()
    return args
cfg = get_attr()
fdv = FinalDataValidator(cfg.subject, verbose = True)
# Check and modify final data, get invalid parts and save history
new_invalid_parts = {}
for k in tqdm(fdv.final_data.keys()):
    invalid_parts = fdv.update_and_mod_invalid_parts(k)
    new_invalid_parts.update(invalid_parts)
# save checked final data and cleared invalid parts
arranged_data_dir = os.path.join('data/wikidata/s14_arranged_final_data', cfg.subject)
os.makedirs(arranged_data_dir, exist_ok=True)
with open(os.path.join(arranged_data_dir, '0.json'), 'w') as f:
    json.dump(fdv.final_data, f)
invalid_parts_dir = os.path.join('data/wikidata/s14_arranged_final_data/0_invalid_parts')
os.makedirs(invalid_parts_dir, exist_ok=True)
invalid_parts_path = os.path.join(invalid_parts_dir, '%s.json'%cfg.subject)
with open(invalid_parts_path, 'w') as f:
    json.dump(new_invalid_parts, f, indent = 4)
