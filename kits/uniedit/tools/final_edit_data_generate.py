from .es_find_entity_tools import ESFindWikidataEntity
from numpy.random._generator import Generator as RNG
from datetime import datetime, timedelta
from typing import Union, Dict, List   
from threading import Thread
from decimal import Decimal
from openai import OpenAI
from queue import Queue
from time import sleep
from tqdm import tqdm
import json, os, re
import pandas as pd 
import numpy as np
import difflib

class FinalEditDataGenerator():
    def __init__(self, api_key = "<DeepSeek-API-Key>", 
            api_url = 'https://api.deepseek.com', model_name = 'deepseek-chat',
            ai_gen_temperature = 0.5, ai_gen_max_tokens = 1000):
        self.esfwe = ESFindWikidataEntity(print_stat = False)
        self.openai_client = OpenAI(api_key = api_key, base_url = api_url) 
        self.model_name = model_name
        self.ai_gen_temperature = ai_gen_temperature
        self.ai_gen_max_tokens = ai_gen_max_tokens
        with open('data/wikidata/monolingual_map.json', 'r') as f: # load monolingual mapping
            self.monolingual_map = json.load(f)
        with open('data/wikidata/prompts/1_edit_sentence_gen/English_prompt.md', 'r') as f:
            self.edit_data_gen_prompt = f.read()
        with open('data/wikidata/prompts/2_single_path_sentence_gen/English_prompt.md', 'r') as f:
            self.single_path_gen_prompt = f.read()
        with open('data/wikidata/prompts/3_double_path_sentence_gen/English_prompt.md', 'r') as f:
            self.double_path_gen_prompt = f.read()

    def get_maxmum_common_substr(self, sentence: str, complete_sub: str) -> str:
        sentence_low = sentence.lower()
        complete_sub = complete_sub.lower()
        sequence_matcher = difflib.SequenceMatcher(None, sentence_low, complete_sub)
        matched = sequence_matcher.find_longest_match(0, len(sentence_low), 0, len(complete_sub))
        if matched.size == 0:
            return ""
        return sentence[matched.a: matched.a + matched.size]
 
    def openai_data_gen(self, content:str, stop_signals = ["<Generation End>"]):
        response = self.openai_client.chat.completions.create(model = self.model_name,  
            messages = [
                {"role": "system", "content": 'You are a reliable data generator.'},
                {"role": "user", "content": content}], stream = False, stop = stop_signals, 
            max_tokens = self.ai_gen_max_tokens, temperature = self.ai_gen_temperature)
        return response.choices[0].message.content

    def openai_dynamic_gen(self, content:str, stop_signals = ["<Generation End>"], print_ = False): 
        response = self.openai_client.chat.completions.create(model = self.model_name,  
            messages = [
                {"role": "system", "content": "You are a reliable data generator."},
                {"role": "user", "content": content},], stream = True, stop = stop_signals,
            max_tokens = self.ai_gen_max_tokens, temperature = self.ai_gen_temperature)
        buffer = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                delta_content = chunk.choices[0].delta.content
                buffer += delta_content
                if print_:
                    print(delta_content, end="", flush=True)
        return buffer

    def get_edit_data_AI_prompt(self, edit_triple:Dict, max_alias = 3):
        content = self.edit_data_gen_prompt
        # [<Head Entity Label>, <Head Entity Description>, [<Alias 1>, <Alias 2>, ...]]
        head_ent = edit_triple['head_entity']
        head_ent_cont = json.dumps([head_ent['label'], head_ent['description'], head_ent['aliases'][:2]])
        content = content.replace('<Head_Entity_Contents>', head_ent_cont)
        # [<Property Label>, <Property Description>]
        prop = edit_triple['property']
        prop_cont = json.dumps([prop['label'], prop['description']])
        content = content.replace('<Relation_Contents>', prop_cont)
        # [<Tail Entity Label>, <Tail Entity Description>, <Is Single Value>, [<Alias 1>, <Alias 2>, ...]]
        tail_ent = edit_triple['tail_entity']
        tail_label = self.get_triple_entity_label(tail_ent, 'tail_entity')
        if tail_ent['datatype'] == 'wikibase-item':
            tail_description = tail_ent['value']['description']
            tail_alias = tail_ent['value']['aliases'][:max_alias]
        else:
            tail_description = tail_ent['datatype'] 
            tail_alias = []
        tail_ent_cont = json.dumps([tail_label, tail_description, tail_alias])
        # tail_ent_cont = json.dumps([tail_label, tail_description, prop['single_value'], tail_alias])
        content = content.replace('<Tail_Entity_Contents>', tail_ent_cont)
        return content
    
    def get_single_path_data_AI_prompt(self, complete_path, neighbor_triples:Dict, 
                                    sample_alias_indxs, rng:RNG):
        '''sample_alias_indxs: make aliases for these indexs, such as ['e'].'''
        label_aliases = {}
        for path_idx in sample_alias_indxs:
            head_sign = neighbor_triples[path_idx]['triple_structure'][0] # like h_e
            if head_sign.split('_')[-1] == path_idx:
                label_aliases[head_sign] = self.sample_triple_entity_value(
                    neighbor_triples[path_idx]['triple']['head_entity'], 'head_entity', rng)
            tail_sign = neighbor_triples[path_idx]['triple_structure'][2] # like t_e
            if tail_sign.split('_')[-1] == path_idx:
                label_aliases[tail_sign] = self.sample_triple_entity_value(
                    neighbor_triples[path_idx]['triple']['tail_entity'], 'tail_entity', rng)
        structure_hops = []
        kcc = "" # Knowledge_Chain_Content
        for i, path in enumerate(complete_path):
            head_sign, prop_sign, tail_sign, direction, _ = path
            if direction == 'b':
                head_sign, tail_sign = tail_sign, head_sign
            path_idx = prop_sign.split('_')[-1]
            head = neighbor_triples[path_idx]['triple']['head_entity']
            prop = neighbor_triples[path_idx]['triple']['property']
            tail = neighbor_triples[path_idx]['triple']['tail_entity']
            kcc += "<Knowledge %s>\n"%(i + 1)
            kcc += '<Head Entity> '
            if head_sign in label_aliases.keys(): 
                subject = label_aliases[head_sign]
            else:
                subject = self.get_triple_entity_label(head, 'head_entity')
            kcc += '"%s"\n'%subject
            kcc += '<Relation> '
            prop_label = self.get_triple_entity_label(prop, 'property')
            kcc += '%s\n'%json.dumps([prop_label, direction == 'b'])
            kcc += '<Tail Entity> '
            if tail_sign in label_aliases.keys(): 
                obj = label_aliases[tail_sign]
            else:
                obj = self.get_triple_entity_label(tail, 'tail_entity')
            kcc += '"%s"'%obj
            structure_hops.append({'subject': subject, 'target': obj, 
                'path': [head_sign, prop_sign, tail_sign, direction, path_idx]})
            if i != len(complete_path) - 1:
                kcc += '\n'
        ai_prompt = self.single_path_gen_prompt.replace('<Knowledge_Chain_Content>', kcc)
        return ai_prompt, structure_hops

    def get_double_path_data_AI_prompt(self, path1_sentence:str, path2_sentence:str, target:str):
        content = self.double_path_gen_prompt
        content = content.replace('<_Cloze_Prefix_1_>', path1_sentence)
        content = content.replace('<_Cloze_Prefix_2_>', path2_sentence)
        content = content.replace('<_Cloze_Result_>', target)
        return content

    def sample_triple_entity_value_structured(self, ent:Dict, ent_type:str, 
                                    exact_value = True, rng:RNG = None):
        '''ent_type: head_entity/property/tail_entity
        exact_value: if False, quantity, time, and globecoordinate will probably 
            sample numerically varied value, in order to get reasoning 
            Generality/Locality data.
        return: 
        1. {'value_type': 'item', 'value': str}
        2. {'value_type': 'string', 'value': str}
        3. {'value_type': 'math', 'value': str}
        4. {'value_type': 'quantity', 'value': str, 'value_scope': one of 
            ['exact', '>', '<', 'error'], 'unit': str (Value unit)}
        5. {'value_type': 'time', 'value': str, 'value_scope': one of 
            ['exact', '>', '<', 'error']}
        6. {'value_type': 'globecoordinate', 'value': str, 'xy': one of ['latitude', 
            'longitude'], 'globe': str, 'value_scope': one of ['exact', '>', '<', 'error']}
        7. {'value_type': 'monolingualtext', 'value': str, 'language': one of ['Chinese', 
            'English', ...]}
        8. None, when situations that do not meet the preset criteria occur.
        '''
        if ent_type == 'head_entity' or ent_type == 'property':
            strs = [ent['label'], *ent['aliases']]
            return {'value_type': 'item', 'value': strs[rng.integers(0, len(strs))]}
        if ent_type != 'tail_entity':
            raise
        # Below for tail entity 
        datatype = ent['datatype']
        ent_value = ent['value']
        if datatype == 'wikibase-item':
            strs = [ent_value['label'], *ent_value['aliases']]
            return {'value_type': 'item', 'value': strs[rng.integers(0, len(strs))]}
        if datatype in ['string', 'math']:
            return {'value_type': datatype, 'value': ent_value}
        # Below for numerical values
        def get_mod_value(num, scope, rand_range, min_rand):
            assert min_rand > 0.02
            if scope == 'exact': 
                return num
            if scope == 'error':
                scope = ['>', '<'][rng.binomial(1, 0.5)]
            num_type = int if isinstance(num, int) else float
            if scope == '>':
                mod_num = num + (min_rand + rand_range * rng.random())
            elif scope == '<':
                mod_num = num - (min_rand + rand_range * rng.random())
            if num_type == float:
                return num_type('%.2f'%mod_num)
            return num_type(mod_num)
        value_scope = 'exact'
        if not exact_value:
            value_scopes = ['exact', '>', '<', 'error']
            value_scope = value_scopes[rng.integers(0, len(value_scopes))]
        if datatype == 'quantity':
            quan_value = eval(ent_value['amount'])
            mod_quan_value = get_mod_value(quan_value, value_scope, quan_value*0.9, 1)
            if not isinstance(ent_value['unit'], str): 
                raise
            unit_label = None
            if 'http' == ent_value['unit'][:4]:
                unit_id = ent_value['unit'].split('/')[-1]
                unit_label = self.esfwe.es_find_entity_by_ids([unit_id], 
                    'all_entity', 1, False)[0]['labels']
            return {'value_type': 'quantity', 'value': str(mod_quan_value), 
                    'value_scope': value_scope, 'unit': unit_label}
        elif datatype == 'time':
            try:
                tm_date, tm_time = ent_value['time'].split('T')
                if tm_time[:8] != '00:00:00':
                    raise
                if tm_date[0] not in ['-', '+']:
                    raise
                year, month, day = tm_date[1:].split('-')
                year, month, day = int(year), int(month), int(day)
                year = - year if tm_date[0] == '-' else year
                tm_precision = ent_value['precision']
                # tm_postfix = 'BC' if BC_date else 'AD'
            except:
                raise
            # precison larger than granularity of a year
            if tm_precision <= 9: 
                tm_precision = 10 ** (9 - tm_precision)
                valid_year_num = int(year // tm_precision)
                mod_valid_year_num = get_mod_value(valid_year_num, value_scope, valid_year_num*1.5, 1)
                year = int(mod_valid_year_num * tm_precision)
                tm_postfix = 'BC' if year < 0 else 'AD'
                value = str(abs(year)) + ' ' + tm_postfix
                return {'value_type': 'time', 'value': value, 'value_scope': value_scope}
            # precison smaller than granularity of a year, like month/day
            if year <= 0: # BC，do not modify time value that precision smaller than year
                if tm_precision == 10: # month
                    tm_postfix = 'BC' if year < 0 else 'AD'
                    return {'value_type': 'time', 'value_scope': 'exact',
                        'value': '%s-%s %s'%(abs(year), month, tm_postfix)}
                elif tm_precision == 11: # day
                    tm_postfix = 'BC' if year < 0 else 'AD'
                    return {'value_type': 'time', 'value_scope': 'exact',
                        'value': '%s-%s-%s %s'%(abs(year), month, day, tm_postfix)}
                else:
                    raise
            else: # AD
                try: # may datetime error
                    if tm_precision == 10: # month
                        mod_day = get_mod_value(0, value_scope, 24 * 30, 32)
                        time_obj = datetime(year, month, 1) + timedelta(mod_day)
                        value = '%s-%s AD'%(time_obj.year, time_obj.month)
                        return {'value_type': 'time', 'value_scope': value_scope, 'value': value}
                    elif tm_precision == 11: # day
                        mod_day = get_mod_value(0, value_scope, 24 * 30, 2)
                        time_obj = datetime(year, month, day) + timedelta(mod_day)
                        value = '%s-%s-%s AD'%(time_obj.year, time_obj.month, time_obj.day)
                        return {'value_type': 'time', 'value_scope': value_scope, 'value': value}
                    else: # do not gen for time precision smaller than a day 
                        raise
                except:
                    raise
        elif datatype in ['globecoordinate', 'globe-coordinate']: # Keep only two decimal places
            try:
                xy = ['latitude', 'longitude'][rng.binomial(1, 0.5)]
                value = float(ent_value[xy])
                mod_value = get_mod_value(value, value_scope, value * 0.5, 0.3)
                blobe_id = ent_value['globe'].split('/')[-1]
                globe_label = self.esfwe.es_find_entity_by_ids([blobe_id], 
                    'all_entity', 1, False)[0]['labels']
                return {'value_type': 'globecoordinate', 'value': '%.2f'%mod_value, 
                    'xy': xy, 'globe': globe_label, 'value_scope': value_scope}
            except:
                raise
        elif datatype == 'monolingualtext': 
            if ent_value['language'] in self.monolingual_map.keys():
                return {'value_type': 'monolingualtext', 'value': ent_value['text'], 
                'language': self.monolingual_map[ent_value['language']]}
            else:
                raise
        raise

    def sample_triple_entity_value(self, ent:Dict, ent_type:str, rng:RNG = None):
        if ent_type in ['property', 'head_entity'] or (ent_type == 'tail_entity' and ent['datatype'] == 'wikibase-item'):
            value = self.sample_triple_entity_value_structured(ent, ent_type, True, rng)
            return value['value']
        else:
            return self.get_triple_entity_label(ent, ent_type)

    def get_triple_entity_label(self, ent:Dict, ent_type:str)->str: 
        '''ent_type: head_entity/property/tail_entity
        return: 
        1. a string type label.
        2. None, when situations that do not meet the preset criteria occur.'''
        if ent_type == 'head_entity' or ent_type == 'property':
            return ent['label']
        if ent_type != 'tail_entity':
            raise
        # Below for tail entity 
        datatype = ent['datatype']
        ent_value = ent['value']
        if datatype == 'wikibase-item':
            return ent_value['label']
        elif datatype in ['string', 'math']:
            return ent_value
        elif datatype == 'quantity':
            if not isinstance(ent_value['unit'], str): 
                raise
            unit_label = None
            if 'http' == ent_value['unit'][:4]:
                unit_id = ent_value['unit'].split('/')[-1]
                unit_label = self.esfwe.es_find_entity_by_ids([unit_id], 
                    'all_entity', 1, False)[0]['labels']
            return_str = '%s'%ent_value['amount'][1:] if ent_value['amount'][0] == '+' else ent_value['amount']
            if unit_label != None:
                return_str += ' %s'%unit_label
            return return_str
        elif datatype == 'time':
            try:
                tm_date, tm_time = ent_value['time'].split('T')
                if tm_time[:8] != '00:00:00':
                    raise
                if tm_date[0] not in ['-', '+']:
                    raise
                year, month, day = tm_date[1:].split('-')
                year, month, day = int(year), int(month), int(day)
                tm_postfix = 'AD' if tm_date[0] == '+' else 'BC'
                tm_precision = ent_value['precision']
            except:
                raise
            if tm_precision <= 9: # precison larger than granularity of a year
                tm_precision = 10 ** (9 - tm_precision)
                year = int(year // tm_precision) * tm_precision
                return '%s %s'%(year, tm_postfix)
            elif tm_precision == 10: # month
                return '%s-%s %s'%(year, month, tm_postfix)
            elif tm_precision == 11: # day
                return '%s-%s-%s %s'%(year, month, day, tm_postfix)
            else:
                raise
        elif datatype in ['globecoordinate', 'globe-coordinate']: # Keep only two decimal places
            try:
                blobe_id = ent_value['globe'].split('/')[-1]
                globe_label = self.esfwe.es_find_entity_by_ids([blobe_id], 
                    'all_entity', 1, False)[0]['labels']
                return_str = '%s: latitude %.2f, longitude %.2f'%(globe_label, 
                        float(ent_value['latitude']), float(ent_value['longitude']))
                return return_str
            except:
                raise
        elif datatype == 'monolingualtext': 
            if ent_value['language'] in self.monolingual_map.keys():
                return "%s: %s"%(self.monolingual_map[ent_value['language']], ent_value['text'])
            else:
                raise
        raise

    def extract_from_ai_return(self, match_str:str, ai_return:str, ):
        matched = re.search(match_str, ai_return)
        if matched:
            return matched.group(1).strip()
        else:
            print(datetime.now(), 'AI return:', ai_return)
            print(datetime.now(), 'Match pattern:', match_str)
            raise ValueError(f"Pattern not found in AI return.")

    def get_final_single_path_data(self, complete_path, neighbor_triples, sample_alias_indxs, rng:RNG):
        ai_prompt, structure_hops = self.get_single_path_data_AI_prompt(
            complete_path, neighbor_triples, sample_alias_indxs, rng)
        ai_return = self.openai_dynamic_gen(ai_prompt)
        one_hops = []
        for i, structure_hop in enumerate(structure_hops):
            head_sign, prop_sign, tail_sign, direction, path_idx = structure_hop['path']
            match_str = '<One-hop Cloze Prefix %d>(.*?)<One-hop Cloze Prefix %d End>'%(i + 1, i + 1)
            prompt = self.extract_from_ai_return(match_str, ai_return)
            match_str = '<One-hop Cloze %d>(.*?)<One-hop Cloze %d End>'%(i + 1, i + 1)
            target = self.extract_from_ai_return(match_str, ai_return)
            subject = self.get_maxmum_common_substr(prompt, structure_hop['subject'])
            one_hop = {
                'path': [head_sign, prop_sign, tail_sign],
                'reverse_in_multi_hop': direction == 'b',
                'prompt': prompt,
                'subject': subject, 
                'target': target,
                'head_entity': neighbor_triples[path_idx]['triple']['head_entity'],
                'property': neighbor_triples[path_idx]['triple']['property'],
                'tail_entity': neighbor_triples[path_idx]['triple']['tail_entity'],
                'single_inverse': neighbor_triples[path_idx]['triple']['single_inverse']
            }
            if direction == 'b':
                match_str = '<R-One-hop Cloze Prefix %d>(.*?)<R-One-hop Cloze Prefix %d End>'%(i + 1, i + 1)
                prompt = self.extract_from_ai_return(match_str, ai_return)
                match_str = '<R-One-hop Cloze %d>(.*?)<R-One-hop Cloze %d End>'%(i + 1, i + 1)
                target = self.extract_from_ai_return(match_str, ai_return)
                one_hop['reversed'] = {
                    'prompt': prompt,
                    'target': target
                }
            one_hops.append(one_hop)
        match_str = '<Multi-hop Cloze Prefix>(.*?)<Multi-hop Cloze Prefix End>'
        prompt = self.extract_from_ai_return(match_str, ai_return)
        match_str = '<Multi-hop Cloze>(.*?)<Multi-hop Cloze End>'
        target = self.extract_from_ai_return(match_str, ai_return)
        return_data = {
            'path_type': 'single',
            'prompt': prompt, 
            'target': target,
            'paths': complete_path, 
            'one_hops': one_hops, 
        }
        return return_data
    
    def get_final_double_path_data(self, complete_paths, neighbor_triples:Dict, 
            sample_alias_indxs:List[str], rng:RNG, path1_data = None, path2_data = None):
        if path1_data == None:
            path1_data = self.get_final_single_path_data(complete_paths[0], neighbor_triples, sample_alias_indxs, rng)
        if path2_data == None:
            path2_data = self.get_final_single_path_data(complete_paths[1], neighbor_triples, sample_alias_indxs, rng)
        ai_prompt = self.get_double_path_data_AI_prompt(path1_data['prompt'], 
            path2_data['prompt'], path1_data['target'])
        ai_return = self.openai_dynamic_gen(ai_prompt)
        match_str = "<Merged Prefix>(.*?)<Merged Prefix End>"
        prompt = self.extract_from_ai_return(match_str, ai_return)
        return_data = {
            'path_type': 'double',
            'prompt': prompt,
            'target': 'Yes',
            'single_path1': path1_data,
            'single_path2': path2_data
        }
        return return_data

    def get_final_edit_data(self, edit_triple:Dict):
        ai_prompt = self.get_edit_data_AI_prompt(edit_triple)
        ai_return = self.openai_dynamic_gen(ai_prompt)
        prompt = self.extract_from_ai_return(r'<Cloze Prefix>(.*?)<Cloze Prefix End>', ai_return)
        target = self.extract_from_ai_return(r'<Cloze>(.*?)<Cloze End>', ai_return)
        subject = self.get_maxmum_common_substr(prompt, edit_triple['head_entity']['label'])
        return_dict = {
            'prompt': prompt, 
            'target': target, 
            'subject': subject, 
            'head_entity': edit_triple['head_entity'],
            'property': edit_triple['property'],
            'tail_entity': edit_triple['tail_entity'],
            'single_inverse': edit_triple['single_inverse'],
        }
        return return_dict
    
    def get_a_final_data(self, a_structured_data:Dict, d1_rng:RNG, 
                         part_of_final_data:Dict = {}):
        '''Assume `a_structured_data`: {
            "edit": {"head_entity": dict, "property": dict, 
                "tail_entity": dict, "single_inverse": bool},
            "generality": {
                "0": {
                    "path_type": one of ["single", "double"],
                    "complete_paths": [[[h_e,r_e,t_e], ...], [if double]], 
                    "neighbor_triples": {
                        "0": {
                            "triple_structure": list,
                            "triple": {"head_entity": dict, "property": ...}
                        }, ...
                    }
                }, ... 
            }
            "locality": {
                "0": {
                    "loc_type": str,
                    "path_type": one of ["single", "double"],
                    "complete_paths": [[[h_e,r_e,t_e], ...], [if double]], 
                    "loc_triples": {
                        "0": {
                            "triple_structure": list,
                            "triple": {"head_entity": dict, "property": ...}
                        }, ...
                    }
                }, ... 
            }
        }'''
        for gd in a_structured_data['generality'].values(): 
            gd['neighbor_triples']['e'] = {
                'triple_structure': ['h_e', 'r_e', 't_e'],
                'triple': a_structured_data['edit']
            } # add edit triple into generality neighbors
        final_data_queue = Queue() 
        def add_edit_final_data():
            if self.has_valid_pt(part_of_final_data, ['edit']):
                put_data = part_of_final_data['edit']
            else:
                put_data = self.get_final_edit_data(a_structured_data['edit'])
            final_data_queue.put(('edit', put_data))
        def add_gen_final_data(idx:str, d2_rng:RNG):
            gen_d = a_structured_data['generality'][idx]
            if self.has_valid_pt(part_of_final_data, ['generality', idx]):
                put_data = part_of_final_data['generality'][idx]
            else:
                if gen_d['path_type'] == 'single':
                    put_data = self.get_final_single_path_data(gen_d['complete_paths'][0], 
                                    gen_d['neighbor_triples'], ['e'], d2_rng)
                else: # double
                    path1_data, path2_data = None, None
                    if self.has_valid_pt(part_of_final_data, ['generality', idx, 'single_path1']):
                        path1_data = part_of_final_data['generality'][idx]['single_path1']
                    if self.has_valid_pt(part_of_final_data, ['generality', idx, 'single_path2']):
                        path2_data = part_of_final_data['generality'][idx]['single_path2']
                    put_data = self.get_final_double_path_data(gen_d['complete_paths'], 
                        gen_d['neighbor_triples'], ['e'], d2_rng, path1_data, path2_data)
            final_data_queue.put(('generality', idx, put_data))
        def add_loc_final_data(idx:str, d2_rng:RNG):
            loc_d = a_structured_data['locality'][idx]
            if self.has_valid_pt(part_of_final_data, ['locality', idx]):
                put_data = part_of_final_data['locality'][idx]
            else:
                put_data = {"loc_type": loc_d['loc_type']}
                if loc_d['path_type'] == 'single':
                    for k, v in self.get_final_single_path_data(loc_d['complete_paths'][0], 
                                loc_d['loc_triples'], [], d2_rng).items():
                        put_data[k] = v
                else:
                    path1_data, path2_data = None, None
                    if self.has_valid_pt(part_of_final_data, ['locality', idx, 'single_path1']):
                        path1_data = part_of_final_data['locality'][idx]['single_path1']
                    if self.has_valid_pt(part_of_final_data, ['locality', idx, 'single_path2']):
                        path2_data = part_of_final_data['locality'][idx]['single_path2']
                    for k, v in self.get_final_double_path_data(loc_d['complete_paths'], 
                                loc_d['loc_triples'], [], d2_rng, path1_data, path2_data).items():
                        put_data[k] = v
            final_data_queue.put(('locality', idx, put_data))
        threads = [Thread(target = add_edit_final_data, args=())]
        for i in a_structured_data['generality'].keys():
            d2_rng = np.random.default_rng(d1_rng.integers(0, 9999999999)) 
            threads.append(Thread(target = add_gen_final_data, args=(i, d2_rng)))
        for i in a_structured_data['locality'].keys():
            d2_rng = np.random.default_rng(d1_rng.integers(0, 9999999999)) 
            threads.append(Thread(target = add_loc_final_data, args=(i, d2_rng)))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        final_data = {'edit': None, 'generality': {}, 'locality': {}}
        while not final_data_queue.empty():
            get_data = final_data_queue.get()
            if get_data[0] == 'edit':
                final_data['edit'] = get_data[1]
            else:
                final_data[get_data[0]][get_data[1]] = get_data[2]
        return final_data
    
    def has_valid_pt(self, d:Dict, keys:List[str] = []):
        for k in keys: 
            if k not in d.keys(): 
                return False
            d = d[k]
        if not isinstance(d, dict): return False
        if 'prompt' not in d.keys() or 'target' not in d.keys(): return False
        if d['prompt'] in [None, ""] or d['prompt'] in [None, ""]: return False
        return True

    def merge_two_final_data(self, fd1:Dict, fd2:Dict):
        # Cover fd2 items to fd1, if they have generated prompt and target in fd2.
        def merge_for_gen_loc(gl, gli):
            gld = fd2[gl][gli]
            if self.has_valid_pt(gld): # has final prompt and target
                if gl not in fd1.keys(): fd1[gl] = {}
                fd1[gl][gli] = gld
            else: # not has final prompt and target, but double path can has a part of results
                if gld['path_type'] == 'double':
                    for path in ['single_path1', 'single_path2']:
                        if self.has_valid_pt(gld, [path]):
                            if gl not in fd1.keys(): fd1[gl] = {}
                            if gli not in fd1[gl].keys(): fd1[gl][gli] = {}
                            fd1[gl][gli][path] = gld[path]
        if self.has_valid_pt(fd2, ['edit']):
            fd1['edit'] = fd2['edit']
        if 'generality' in fd2.keys():
            for gi in fd2['generality'].keys():
                merge_for_gen_loc('generality', gi)
        if 'locality' in fd2.keys():
            for li in fd2['locality'].keys():
                merge_for_gen_loc('locality', li)
        return fd1

    def load_and_merge_final_data(self, final_subject_data_dir):
        final_data = {}
        file_names = ['%s.json'%i for i in sorted([int(n.split('.')[0]) for n in os.listdir(final_subject_data_dir)])]
        for data_name in file_names:
            with open(os.path.join(final_subject_data_dir, data_name), 'r') as f:
                a_batch_final_data = json.load(f)
                for fdi, fd in a_batch_final_data.items():
                    if fdi in final_data.keys():
                        final_data[fdi] = self.merge_two_final_data(final_data[fdi], fd)
                    else:
                        final_data[fdi] = fd
        return final_data
    
    def load_structured_data(self, structured_sub_data_dir):
        all_structured_data = {}
        for data_name in os.listdir(structured_sub_data_dir):
            with open(os.path.join(structured_sub_data_dir, data_name), 'r') as f: 
                all_structured_data.update(json.load(f))
        return all_structured_data

    def is_complete_final_data(self, final_data:Dict, struct_data:Dict):
        if not self.has_valid_pt(final_data, ['edit']):
            return False
        for gi in struct_data['generality'].keys():
            if not self.has_valid_pt(final_data, ['generality', gi]):
                return False
        for li in struct_data['locality'].keys():
            if not self.has_valid_pt(final_data, ['locality', li]):
                return False
        return True

    def generate_all_final_data(self, data_subject, max_data_gen = 9999999, 
                    save_every = 4, max_thread = 200, random_seed = 1234,
                    proc_start_time = datetime.strptime('00:45', '%H:%M'), 
                    proc_end_time = datetime.strptime('08:15', '%H:%M')):
        '''A structured data: {
            "edit": {<a_triple>},
            "generality":{
                "0": {
                    "path_type": "single/double", 
                    "simple_paths": [...], "complete_paths": [...], 
                    "neighbor_triples": {
                        "0": {"triple_structure": [...], "triple": {<a_triple>}, 
                        ...
                    }
                }, ...
            }
            "locality": {
                "0": {
                    "loc_type": "head/prop/tail of edit",
                    "path_type": "single/double", 
                    "simple_paths": [...], "complete_paths": [...], 
                    "loc_triples": {
                        "l": {"triple_structure": [...], "triple": {<a_triple>}, 
                        "0": {"triple_structure": [...], "triple": {<a_triple>}, 
                        ...
                    }
                }
            }
        }
        A final data: {
            "edit": {"prompt": "", "target": "", "subject": "", **{<a_triple>}},
            "generality":{
                "0": { # Single path for example.
                    "prompt": "", "target": "", "path_type": "single", "paths": [...]
                    "one_hops": [
                        {"prompt": "", "target": "", "subject": "", "path": [...], "reverse_in_multi_hop": bool,  **{<a_triple>}}, ...
                    ]
                }, 
                "1": { # Double path for example, based on single paths.
                    "prompt": "", "target": "", "path_type": "double"
                    "single_path1": {
                        "prompt": "", "target": "", "path_type": "single",
                        "paths": [...], "one_hops": [...]
                    },
                    "single_path2": {
                        "prompt": "", "target": "", "path_type": "single", 
                        "paths": [...], "one_hops": [...]
                    }
                }, 
                ...
            }
            "locality": {... # Like generality
            }
        }'''
        meta_rng = np.random.default_rng(random_seed)
        # Load all structured data
        structured_data_dir = os.path.join('data/wikidata/s12_structured_data', data_subject)
        all_structured_data = self.load_structured_data(structured_data_dir)
        # load existing final data, to avoid duplicate generation
        final_data_dir = os.path.join('data/wikidata/s13_final_data', data_subject)
        if not os.path.exists(final_data_dir):
            os.makedirs(final_data_dir) 
        exist_final_data = self.load_and_merge_final_data(final_data_dir)
        # start thread of saving final data
        if_all_thread_end = False
        final_data_queue = Queue()
        def save_data_func():
            while True:
                sleep(1)
                if not if_all_thread_end:
                    if final_data_queue.qsize() < save_every:
                        continue
                    save_len = save_every
                else:
                    if final_data_queue.qsize() == 0:
                        break
                    save_len = final_data_queue.qsize()
                save_data = {}
                for i in range(save_len):
                    k, a_final_data = final_data_queue.get()
                    save_data[k] = a_final_data
                for i in range(9999999999):  
                    save_path = os.path.join(final_data_dir, '%s.json'%i)
                    if not os.path.exists(save_path):
                        with open(save_path, 'w') as f:
                            json.dump(save_data, f)
                        break
        save_data_thread = Thread(target = save_data_func)
        save_data_thread.start()  
        # start threads to get final data
        def thread_get_a_final_data(k, d1_rng): 
            a_final_data = self.get_a_final_data(all_structured_data[k], d1_rng,
                exist_final_data[k] if k in exist_final_data.keys() else {})
            final_data_queue.put((k, a_final_data))
            print(datetime.now(), "Generate and push %s"%k, flush = True)
        data_ks = list(all_structured_data.keys())
        meta_rng.shuffle(data_ks)
        gen_threads = []
        for k in tqdm(data_ks[:max_data_gen], 'Adding generation threads'): 
            if k in exist_final_data.keys() and self.is_complete_final_data(exist_final_data[k], all_structured_data[k]):
                print(datetime.now(), "Final data %s is already completed, skip."%k, flush = True)
                continue
            while not if_can_implement(proc_start_time, proc_end_time):
                print(datetime.now(), "It's not time for execution.", flush = True)
                sleep(60)
            while len(gen_threads) >= max_thread:
                gen_threads = [t for t in gen_threads if t.is_alive()]
                sleep(1)
            d1_rng = np.random.default_rng(meta_rng.integers(0, 9999999999))
            thread = Thread(target = thread_get_a_final_data, args=(k, d1_rng))
            thread.start()
            gen_threads.append(thread)
        for thread in gen_threads:
            thread.join()
        if_all_thread_end = True
        save_data_thread.join()
        print(datetime.now(), "Finished.") 


def if_can_implement(start_time:datetime, end_time:datetime):
    current_time = datetime.now().time()
    if current_time >= start_time.time() and current_time <= end_time.time():
        return True
    return False

