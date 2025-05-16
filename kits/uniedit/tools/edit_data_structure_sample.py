#%% sample edit triple and corresponding generality/locality triples. Used after step 8.
from .es_find_entity_tools import ESFindWikidataEntity
from numpy.random._generator import Generator as RNG
from typing import Union, Dict, List, Set
from datetime import datetime, timedelta
from colorama import Fore, Style
from collections import Counter
from decimal import Decimal
from pprint import pprint
from tqdm import tqdm
from time import time
import json, os, re
import pandas as pd
import numpy as np
import math, difflib

class EditDataStructureSampler():
    def __init__(self, subject = 'biology', ent_min_claim_n = 2):
        self.esfwe = ESFindWikidataEntity(print_stat = False)
        self.ent_min_claim_n = ent_min_claim_n
        # load filtered properties
        filtered_props = pd.read_csv('data/wikidata/property/filtered_property.csv')
        self.filtered_props = {prop['id']:[prop['label'], prop['# item pages']] for prop in filtered_props.iloc}
        # load filtered entity ids
        print('Loading filtered entity ids...')
        with open('data/wikidata/filtered_entity/only_id.json', 'r') as f:
            self.filtered_entid_list = json.load(f)
            self.filtered_entid_set = set(self.filtered_entid_list)
        # load subjects & subject sample probability
        self.subject = subject
        subject_path = os.path.join('data/wikidata/s8_subject_entity/', '%s.csv'%subject)
        self.sub_ent = []
        retr_scores, share_sub_words_ns = [], []
        sub_ents = pd.read_csv(subject_path).to_dict('records')
        for ent in tqdm(sub_ents, 'Loading subject entity data'):
            ent['claims'] = eval(ent['claims'])
            clms_n = len([1 for clm in ent['claims'] if clm in self.filtered_props])
            if clms_n >= ent_min_claim_n:
                ent['share_sub_words'] = set(eval(ent['share_sub_words']))
                ent['non_sub_words'] = set(eval(ent['non_sub_words']))
                self.sub_ent.append(ent)
                retr_scores.append(ent['retr_score'])
                share_sub_words_ns.append(ent['share_sub_words_n'])
        self.edit_entity_sample_weight = np.array(retr_scores) * np.array(share_sub_words_ns)

    def build_property_to_entity_map(self):
        self.propt_to_ent = {p:[] for p in self.filtered_props.keys()}
        for ent in tqdm(self.sub_ent, 'Building property to entity mapping'):
            for clm in ent['claims']:
                if clm in self.propt_to_ent:
                    self.propt_to_ent[clm].append(ent['id'])

    def is_item_value_with_id(self, item_value:Union[dict, float, int, str]):
        # check if item_value is an item within id
        return isinstance(item_value, dict) and 'id' in item_value.keys()

    def random_filtered_entid(self, rng:RNG):
        return self.filtered_entid_list[rng.integers(0, len(self.filtered_entid_list))]

    def get_details_for_ents_with_id(self, items_obj:Union[Dict, List], add_tqdm = True):
        # traverse items_obj and get label/description/aliases fot them
        def get_all_ent_ids_and_value_dict(in_obj:Union[Dict, List]):
            ids, value_dicts = [], []
            if isinstance(in_obj, dict):
                if 'id' in in_obj.keys() and 'label' not in in_obj.keys():
                    ids.append(in_obj['id'])
                    value_dicts.append(in_obj)
                for v in in_obj.values():
                    extra_ids, extra_value_dicts = get_all_ent_ids_and_value_dict(v)
                    ids.extend(extra_ids)
                    value_dicts.extend(extra_value_dicts)
            elif isinstance(in_obj, list):
                for l in in_obj:
                    extra_ids, extra_value_dicts = get_all_ent_ids_and_value_dict(l)
                    ids.extend(extra_ids)
                    value_dicts.extend(extra_value_dicts)
            return ids, value_dicts
        ids, value_dicts = get_all_ent_ids_and_value_dict(items_obj)
        ids = set(ids)
        detail_ents = {e['id']:e for e in self.esfwe.es_find_entity_by_ids(ids, 
            'all_entity', add_tqdm = add_tqdm)}
        for et in value_dicts:
            i = et['id']
            if i not in detail_ents: # entity id not found in exisiting entities
                continue
            et['label'] = detail_ents[i]['labels']
            et['description'] = detail_ents[i]['descriptions']
            et['aliases'] = eval(detail_ents[i]['aliases'])
            if not isinstance(et['aliases'], list): 
                et['aliases'] = []
            if 'numeric-id' in et.keys(): et.pop('numeric-id')
            if 'entity-type' in et.keys(): et.pop('entity-type')
    
    def print_triple(self, triple:dict):
        print("(%s%s%s, %s%s%s, %s%s%s)" % (
            Fore.RED, triple['head_entity']['label'], Fore.RESET,
            Fore.GREEN, triple['property']['label'], Fore.RESET,
            Fore.BLUE,
            triple['tail_entity']['value']['label'] if self.is_item_value_with_id(triple['tail_entity']['value'])
            else triple['tail_entity']['value'],
            Fore.RESET
        ))
        if 'counterfactual_sample' in triple:
            print('Counterfactual tail:', triple['counterfactual_sample']['tail_entity']['value']['label'] 
                if self.is_item_value_with_id(triple['counterfactual_sample']['tail_entity']['value']) else 
                triple['counterfactual_sample']['tail_entity']['value'])

    def find_triples_given_heads_and_tail_id(self, head_ents:List, tail_id:str):
        results = []
        for ent in tqdm(head_ents, 'Finding triples'):
            if isinstance(ent['claims'], str):
                ent['claims'] = eval(ent['claims'])
            for p in ent['claims']:
                for clm in ent['claims'][p]:
                    if clm['mainsnak']['snaktype'] != 'value': continue
                    if clm['mainsnak']['datatype'] != 'wikibase-item': continue
                    if clm['mainsnak']['datavalue']['value']['id'] == tail_id:
                        triple = {'head_entity': {'id':ent['id']}, 'property': {'id': p}, 
                                'tail_entity': {'id': tail_id}}
                        results.append(triple)
        return results

    def if_triple_info_leak(self, triple: Dict, leak_thres = 0.6):
        # tail label should not in head label/description/aliases
        head, tail = triple['head_entity'], triple['tail_entity']
        tail_datatype = tail['datatype']
        tail_value = tail['value']
        check_values = []
        if tail_datatype == 'wikibase-item':
            check_values.append(tail_value['label'])
        elif tail_datatype in ['string', 'math']:
            check_values.append(tail_value)
        elif tail_datatype == 'quantity':
            check_values.append(str(eval(tail_value['amount'])))
        elif tail_datatype == 'time':
            if tail_value['precision'] > 11:
                return False
            tm_date, _ = tail_value['time'].split('T') # tm_date, tm_time
            if tm_date[0] not in ['-', '+']: raise
            year, _, _ = tm_date[1:].split('-') # year, month, day
            check_values.append(year)
        elif tail_datatype in ['globecoordinate', 'globe-coordinate']:
            latitude, longitude = float(tail_value['latitude']), float(tail_value['longitude'])
            check_values.append(str(math.floor(latitude * 100) / 100))
            check_values.append(str(math.floor(longitude * 100) / 100))
        elif tail_datatype == 'monolingualtext':
            check_values.append(tail_value['text'])
        for s in [head['label'], head['description'], *head['aliases']]:
            s = s.lower()
            for cv in check_values:
                sequence_matcher = difflib.SequenceMatcher(None, cv.lower(), s)
                if sequence_matcher.find_longest_match().size/len(cv) > leak_thres:
                    return True 
        return False
    
    def sample_triple_given_head(self, head_ent:dict, tail_inside_filtered = True,
            property_inside_filtered = True, except_tail_ids:Set = set(), rng:RNG = None,
            prop_samp_minor_w:Dict[str, int] = None, drop_info_leak = True):
        ''' Sample a triple given a head entity. `tail_inside_filtered` and `property_inside_filtered`
        control whether the sampled tail and property should be in the filterd sets.
        `except_tail_ids`: the tails should not in the triple.
        `prop_samp_minor_w`: the property sample weights, like {'P31': 5, 'P**': 2}. 
        The sampling probability of propertes is proportional to the reciprocal of the weights.''' 
        triple = {'head_entity': {}, 'property': {}, 'tail_entity': {}, 'qualifiers': {}}
        if 'claims' not in head_ent.keys(): 
            head_ent = self.esfwe.es_find_entity_by_ids([head_ent['id']], 'all_entity')[0]
        if type(head_ent['claims']) == str: 
            head_ent['claims'] = eval(head_ent['claims']) 
        # sample & add tail entity/qualifiers/property/head entity
        prop_ids = list(head_ent['claims'].keys()) 
        if prop_samp_minor_w == None:
            rng.shuffle(prop_ids)
        else:
            prop_sample_ws = [1/prop_samp_minor_w[i] if i in prop_samp_minor_w.keys() else 1 
                            for i in prop_ids]
            weight_sum = sum(prop_sample_ws)
            prop_p = [w/weight_sum for w in prop_sample_ws]
            prop_ids = rng.choice(prop_ids, len(prop_ids), False, prop_p, shuffle = False)
        for sampled_prop_id in prop_ids:
            if property_inside_filtered and sampled_prop_id not in self.filtered_props:
                continue
            sampled_prop = head_ent['claims'][sampled_prop_id]
            tail_entity_orders = list(range(len(sampled_prop))) 
            rng.shuffle(tail_entity_orders)
            for tail_entity_order in tail_entity_orders:
                tail_entity = sampled_prop[tail_entity_order]
                if tail_entity['mainsnak']['snaktype'] != 'value':
                    continue
                if self.is_item_value_with_id(tail_entity['mainsnak']['datavalue']['value']):
                    tail_id = tail_entity['mainsnak']['datavalue']['value']['id']
                    if tail_inside_filtered and tail_id not in self.filtered_entid_set:
                        continue
                    if tail_id in except_tail_ids:
                        continue
                # add head entity
                triple['head_entity'] = {
                    'id': head_ent['id'], 
                    'label': head_ent['labels'], 
                    'description': head_ent['descriptions'], 
                    'aliases': eval(head_ent['aliases']), 
                }
                try:
                    triple['head_entity']['P31'] = [p31v['mainsnak']['datavalue']['value']['id'] 
                        for p31v in head_ent['claims']['P31'] if p31v['mainsnak']['datatype'] == 'wikibase-item']
                except:
                    triple['head_entity']['P31'] = []
                if not isinstance(triple['head_entity']['aliases'], list): 
                    triple['head_entity']['aliases'] = []
                # add property
                triple['property'] = {
                    'id': sampled_prop_id,
                    'single_value': len(sampled_prop) == 1,
                }
                # add tail entity
                triple['tail_entity'] = {
                    'order_in_property': tail_entity_order,
                    'datatype': tail_entity['mainsnak']['datatype'],
                    'value': tail_entity['mainsnak']['datavalue']['value'],
                }
                # Judge if the information of the tail entity is leaked in the head entity
                if drop_info_leak:
                    self.get_details_for_ents_with_id(triple, False)
                    if self.if_triple_info_leak(triple):
                        continue
                # add tail entity qualifiers
                if 'qualifiers' in tail_entity.keys():
                    for p, ents in tail_entity['qualifiers'].items():
                        try:
                            triple['qualifiers'][p] = {
                                'property': {'id': p},
                                'values': [{'datatype': e['datatype'], 
                                    'value': e['datavalue']['value']} for e in ents] 
                            }
                        except: pass
                return triple
        return None

    def sample_triple_given_tail(self, tail_ent:Dict, head_inside_filtered = True, 
            property_inside_filtered = True, es_match_size = 256, except_head_ids:Set = set(), 
            matched_ents_cache = None, return_matched_ents = False, rng:RNG = None,
            drop_info_leak = True):
        # the property and the extra entity should inside the filtered properties and entities
        triple = {'head_entity': {}, 'property': {}, 'tail_entity': {}, 'qualifiers': {}}
        tail_ent_id = tail_ent['id']
        if matched_ents_cache != None:
            matched_ents = matched_ents_cache
        else:
            _, _, _, matched_ents = self.esfwe.es_multi_fields_match(
                tail_ent_id, es_match_size, 'all_entity', ['claims'], False)
        rng.shuffle(matched_ents)
        for head_ent in matched_ents:
            if head_inside_filtered and head_ent['id'] not in self.filtered_entid_set:
                continue
            if head_ent['id'] in except_head_ids:
                continue
            if type(head_ent['claims']) == str: 
                head_ent['claims'] = eval(head_ent['claims']) 
            clm_keys = list(head_ent['claims'].keys())
            rng.shuffle(clm_keys)
            for prop_id in clm_keys:
                if property_inside_filtered and prop_id not in self.filtered_props:
                    continue
                tail_entity_orders = list(range(len(head_ent['claims'][prop_id])))
                rng.shuffle(tail_entity_orders)
                for tail_entity_order in tail_entity_orders:
                    prop_v = head_ent['claims'][prop_id][tail_entity_order]
                    if prop_v['mainsnak']['snaktype'] != 'value' or prop_v[
                        'mainsnak']['datatype'] != 'wikibase-item':
                        continue
                    if prop_v['mainsnak']['datavalue']['value']['id'] == tail_ent_id:
                        # add head entity
                        triple['head_entity'] = {
                            'id': head_ent['id'], 
                            'label': head_ent['labels'], 
                            'description': head_ent['descriptions'], 
                            'aliases': eval(head_ent['aliases']), 
                        }
                        if not isinstance(triple['head_entity']['aliases'], list): 
                            triple['head_entity']['aliases'] = []
                        # add property
                        triple['property'] = {
                            'id': prop_id,
                            'single_value': len(head_ent['claims'][prop_id]) == 1,
                        }
                        # add tail entity
                        triple['tail_entity'] = {
                            'order_in_property': tail_entity_order,
                            'datatype': prop_v['mainsnak']['datatype'],
                            'value': {'id': tail_ent['id']}
                        }
                        if 'label' in tail_ent.keys():
                            triple['tail_entity']['value']['label'] = tail_ent['label']
                            triple['tail_entity']['value']['description'] = tail_ent['description']
                            triple['tail_entity']['value']['aliases'] = tail_ent['aliases']
                        # Judge if the information of the tail entity is leaked in the head entity
                        if drop_info_leak:
                            self.get_details_for_ents_with_id(triple, False)
                            if self.if_triple_info_leak(triple):
                                continue
                        # add tail entity qualifiers
                        if 'qualifiers' in prop_v.keys():
                            for p, ents in prop_v['qualifiers'].items():
                                try:
                                    triple['qualifiers'][p] = {
                                        'property': {'id': p},
                                        'values': [{'datatype': e['datatype'], 
                                            'value': e['datavalue']['value']} for e in ents] 
                                    }
                                except: pass
                        if return_matched_ents:
                            return triple, matched_ents
                        return triple
        if return_matched_ents:
            return None, matched_ents
        return None

    def sample_triple_given_property(self, prop_ent:Dict, head_inside_filtered = True, 
            tail_inside_filtered = True, es_match_size = 512, matched_ents_cache = None, 
            return_matched_ents = False, rng:RNG = None, drop_info_leak = True):
        # the property and the extra entity should inside the filtered properties and entities
        triple = {'head_entity': {}, 'property': {}, 'tail_entity': {}, 'qualifiers': {}}
        prop_id = prop_ent['id']
        if matched_ents_cache != None:
            matched_ents = matched_ents_cache
        else:
            _, _, _, matched_ents = self.esfwe.es_multi_fields_match(
                prop_id, es_match_size, 'all_entity', ['claims'], False) 
        rng.shuffle(matched_ents)
        for head_ent in matched_ents:
            if head_inside_filtered and head_ent['id'] not in self.filtered_entid_set:
                continue
            if type(head_ent['claims']) == str: 
                head_ent['claims'] = eval(head_ent['claims']) 
            if prop_id not in head_ent['claims'].keys():
                continue
            prop_vs = head_ent['claims'][prop_id]
            tail_entity_orders = list(range(len(prop_vs)))
            rng.shuffle(tail_entity_orders)
            for tail_entity_order in tail_entity_orders:
                tail_entity = prop_vs[tail_entity_order]
                if tail_entity['mainsnak']['snaktype'] != 'value':
                    continue
                if self.is_item_value_with_id(tail_entity['mainsnak']['datavalue']['value']):
                    tail_id = tail_entity['mainsnak']['datavalue']['value']['id']
                    if tail_inside_filtered and tail_id not in self.filtered_entid_set:
                        continue
                # add head entity
                triple['head_entity'] = {
                    'id': head_ent['id'], 
                    'label': head_ent['labels'], 
                    'description': head_ent['descriptions'], 
                    'aliases': eval(head_ent['aliases']), 
                }
                try:
                    triple['head_entity']['P31'] = [p31v['mainsnak']['datavalue']['value']['id'] 
                        for p31v in head_ent['claims']['P31'] if p31v['mainsnak']['datatype'] == 'wikibase-item']
                except:
                    triple['head_entity']['P31'] = []
                if not isinstance(triple['head_entity']['aliases'], list): 
                    triple['head_entity']['aliases'] = []
                # add property
                triple['property'] = {
                    'id': prop_id,
                    'single_value': len(prop_vs) == 1,
                }
                # add tail entity
                triple['tail_entity'] = {
                    'order_in_property': tail_entity_order,
                    'datatype': tail_entity['mainsnak']['datatype'],
                    'value': tail_entity['mainsnak']['datavalue']['value'],
                }
                # Judge if the information of the tail entity is leaked in the head entity
                if drop_info_leak:
                    self.get_details_for_ents_with_id(triple, False)
                    if self.if_triple_info_leak(triple):
                        continue
                # add tail entity qualifiers
                if 'qualifiers' in tail_entity.keys():
                    for p, ents in tail_entity['qualifiers'].items():
                        try:
                            triple['qualifiers'][p] = {
                                'property': {'id': p},
                                'values': [{'datatype': e['datatype'], 
                                    'value': e['datavalue']['value']} for e in ents] 
                            }
                        except: pass
                if return_matched_ents:
                    return triple, matched_ents
                return triple
        if return_matched_ents:
            return None, matched_ents
        return None

    def sample_head_entity_for_edit(self, sample_n:int, rng:RNG, sample_batch = 30,
            max_label_len = 10, non_sub_words_hit_reject_stren = 1,
            sub_words_hit_reject_stren = 0.2, reject_base = 1.05, sample_scope = 4)->List[str]: 
        ''' This sample function everytime sample a batch of entities, and uses 
            splitted words of sampled entities' label and description to intervene 
            the sampling probability of remain entities. '''
        # sample head entities, skip entities whose description words repeated
        max_description = {'clinical trial': 1000, 'chemical compound': sample_n * sample_scope//2}
        now_description_n = {}
        sample_pool_scale = int(sample_n * sample_scope)
        indx_pool = []
        for i in np.argsort(self.edit_entity_sample_weight)[::-1]:
            ent = self.sub_ent[i]
            ent['label_len'] = len(ent['labels'].split(' '))
            if ent['label_len'] > max_label_len:
                continue
            ent_desc = ent['description']
            if ent_desc in max_description.keys():
                if ent_desc not in  now_description_n.keys():
                    now_description_n[ent_desc] = 0
                if now_description_n[ent_desc] > max_description[ent_desc]:
                    continue
                now_description_n[ent_desc] += 1
            indx_pool.append(i)
            if len(indx_pool) >= sample_pool_scale:
                break
        sample_ents_pool = [self.sub_ent[i] for i in indx_pool]
        length_punish = np.array([1/((l-max_label_len//2+1)**2) if l > max_label_len//2 else 1 for l in 
                                  [self.sub_ent[i]['label_len'] for i in indx_pool]])
        extra_punish = np.array([0.01 if 'clinical trial' in self.sub_ent[i]['description'] else 1
            for i in indx_pool]) # extra punish to "clinical trial" description, which is too many
        ini_sample_ws = np.array([self.edit_entity_sample_weight[i] for i in indx_pool
                                  ]) * length_punish * extra_punish
        now_sample_ws = ini_sample_ws.copy()
        # Words map to entity index
        sub_words_map_ent_idx = {}
        non_sub_words_map_ent_idx = {}
        for i in tqdm(range(len(sample_ents_pool)), 'Building map'):
            ent = sample_ents_pool[i]
            for w in ent['share_sub_words']:
                if w not in sub_words_map_ent_idx.keys():
                    sub_words_map_ent_idx[w] = set()
                sub_words_map_ent_idx[w].add(i)
            for w in ent['non_sub_words']:
                if w not in non_sub_words_map_ent_idx.keys():
                    non_sub_words_map_ent_idx[w] = set()
                non_sub_words_map_ent_idx[w].add(i)
        sample_indx_pool = list(range(len(sample_ents_pool)))
        exclude_hit_words = set(['of', 'and', 'by', 'is', 'in', 'are', 'on', 
            'the', 'a', 'an', 'for', 'to', 'that', 'at', 'as', 'with', 'from', 
            'which'])
        sub_words_hits = {}
        non_sub_words_hits = {} 
        sampled_ent_ids = set()
        sampled_idexs = set() # index for the sample entity pool, not for self.sub_ent
        sample_head_bar = tqdm(total=min(len(sample_ents_pool), sample_n), desc='Head entity sampling')
        while True:
            sample_indx_pool = list(set(sample_indx_pool) - sampled_idexs)
            now_smaple_ws_rmd = now_sample_ws[sample_indx_pool] 
            print(now_smaple_ws_rmd)
            now_sampled_indx = rng.choice(sample_indx_pool, size = sample_batch, 
                        replace = False, p = now_smaple_ws_rmd / sum(now_smaple_ws_rmd))
            for nsi in now_sampled_indx:
                sampled_ent = sample_ents_pool[nsi]
                sampled_ent_ids.add(sampled_ent['id'])
                sampled_idexs.add(nsi)
                sample_head_bar.update(1)
                # update sample weights for entotoes in index pool
                ents_idx_need_update_p = set()
                for w in sampled_ent['non_sub_words']:
                    if w in exclude_hit_words: continue
                    if w not in non_sub_words_hits.keys():
                        non_sub_words_hits[w] = 0
                    non_sub_words_hits[w] += 1 
                    ents_idx_need_update_p.update(non_sub_words_map_ent_idx[w])
                for w in sampled_ent['share_sub_words']:
                    if w in exclude_hit_words: continue
                    if w not in sub_words_hits.keys():
                        sub_words_hits[w] = 0
                    sub_words_hits[w] += 1
                    ents_idx_need_update_p.update(sub_words_map_ent_idx[w])
                for i in ents_idx_need_update_p:
                    ent = sample_ents_pool[i]
                    non_sub_words_hit_cof = sum([non_sub_words_hits[w] if w in non_sub_words_hits.keys() 
                                                else 0 for w in ent['non_sub_words']])
                    sub_words_hit_cof = sum([sub_words_hits[w] if w in sub_words_hits.keys() 
                                            else 0 for w in ent['share_sub_words']])
                    reject_cof = (non_sub_words_hit_cof * non_sub_words_hit_reject_stren + 
                        sub_words_hit_cof * sub_words_hit_reject_stren
                        ) / (ent['non_sub_words_n'] + ent['share_sub_words_n']) # [0, +infty]
                    now_sample_ws[i] = ini_sample_ws[i] * ((1/reject_base) ** reject_cof)
                    # now_sample_ws[i] = ini_sample_ws[i] / ((1 + reject_cof)**reject_base)
                if len(sampled_ent_ids) >= min(len(sample_ents_pool), sample_n):
                    break
            if len(sampled_ent_ids) >= min(len(sample_ents_pool), sample_n):
                break 
        return list(sampled_ent_ids)

    def sample_edit_triples_given_head_ids(self, sampled_ent_ids:List[str], 
            tail_inside_filtered = True, property_inside_filtered = True, 
            error_skip = True, rng:RNG = None, restrict_prop_over_samp = True, 
            prop_restrict_coef:Dict = {"P279": 100}, max_try_each_head = 3):
        ''' 
        `restrict_prop_over_samp`: if True, the sampling probability of properties 
        will be negatively correlated with its sampling frequency.
        `prop_restrict_coef`: like {"P279": 10}, means property P279's initialized 
        sample number which will negatively influence its subsequent sampling. '''
        # sample edit triples
        head_ents = self.esfwe.es_find_entity_by_ids(
            list(sampled_ent_ids), 'all_entity', add_tqdm = True)
        prop_samp_minor_w = {} if restrict_prop_over_samp else None
        edit_triples = []
        for e in tqdm(head_ents, 'Samplig edit triples'):
            for _ in range(max_try_each_head):
                try: 
                    triple = self.sample_triple_given_head(e, tail_inside_filtered, 
                        property_inside_filtered, rng = rng, prop_samp_minor_w = prop_samp_minor_w)
                    self.get_details_for_ents_with_id(triple, add_tqdm = False)
                    if triple != None: 
                        if restrict_prop_over_samp:
                            prop_id = triple['property']['id']
                            if prop_id not in prop_samp_minor_w:
                                prop_samp_minor_w[prop_id] = 1
                            if prop_id not in prop_restrict_coef.keys():
                                prop_restrict_coef[prop_id] = 0
                            prop_restrict_coef[prop_id] += 1
                            prop_samp_minor_w[prop_id] = 2 ** min(100, prop_restrict_coef[prop_id])
                        edit_triples.append(triple)
                        break
                except: 
                    if error_skip: 
                        pass
                    else: 
                        raise
        return edit_triples 

    def sample_counterfactual_tail(self, edit_triple:dict, search_batch = 256, 
                                   max_search = 512, rng:RNG = None):
        '''Sample a counterfactual tail in subject entities, which has a triple 
            whose property is same with `edit_triple`'s, and head entity share 
            the same `instance of` property with `edit_triple`'s. In addition, 
            the counterfactual tail's data type should be same with `edit_triple`'s, 
            and its value should not be same with `edit_triple`'s. '''
        if not hasattr(self, 'propt_to_ent'): self.build_property_to_entity_map()
        triple_prop_id = edit_triple['property']['id']
        if triple_prop_id == 'P31':
            return False
        head_instance = edit_triple['head_entity']['P31'][rng.integers(
            0, len(edit_triple['head_entity']['P31']))]
        tail_ent_data_type = edit_triple['tail_entity']['datatype']
        tail_ent_value = edit_triple['tail_entity']['value']
        ents_share_p = self.propt_to_ent[triple_prop_id]
        idxs = list(range(len(ents_share_p)))
        rng.shuffle(idxs)
        def get_counterfact_ent(semi_sim_ent):
            # similar entity should have the same class as head entity
            is_sim = False
            if 'P31' not in semi_sim_ent['claims'].keys(): return None
            for p31v in semi_sim_ent['claims']['P31']: 
                try:
                    if p31v['mainsnak']['datavalue']['value']['id'] == head_instance:
                        is_sim = True
                        break 
                except:
                    pass
            if not is_sim: return None
            # similar entity should have the same datatype of the property as tail entity
            prop_vs = semi_sim_ent['claims'][triple_prop_id]
            orders = list(range(len(prop_vs)))
            rng.shuffle(orders)
            for r in orders:
                if prop_vs[r]['mainsnak']['datatype'] == tail_ent_data_type:
                    if self.is_item_value_with_id(prop_vs[r]['mainsnak']['datavalue']['value']):
                        if prop_vs[r]['mainsnak']['datavalue']['value']['id'] == tail_ent_value['id']:
                            continue
                    counterfact_ent = {}
                    counterfact_ent['head_similar_entity'] = {
                        'id': semi_sim_ent['id'],
                        'label': semi_sim_ent['labels'],
                        'description': semi_sim_ent['descriptions'],
                        'P31': head_instance
                    }
                    counterfact_ent['property'] = {
                        'id': triple_prop_id,
                        'single_value': len(prop_vs) == 1
                    }
                    counterfact_ent['tail_entity'] = {
                        'order_in_property': r,
                        'datatype': tail_ent_data_type,
                        'value': prop_vs[r]['mainsnak']['datavalue']['value'],
                    }
                    return counterfact_ent
            return None
        edit_triple['counterfactual_sample'] = None
        for i in range(0, min(max_search, len(ents_share_p)), search_batch):
            ids = [ents_share_p[j] for j in idxs[i:i+search_batch]]
            ents = self.esfwe.es_find_entity_by_ids(ids, 'all_entity')
            for e in ents:
                e['claims'] = eval(e['claims'])
                counterfactual_sample = get_counterfact_ent(e)
                if counterfactual_sample != None: 
                    edit_triple['counterfactual_sample'] = counterfactual_sample
                    return True
        return False

    def sample_multi_hop_neighbor(self, this_triple:dict, this_triple_sign = 'e', 
            neighbor_n = 2, allow_ent_repeat = False, ent_inside_filtered = True, 
            property_inside_filtered = True, rng:RNG = None):
        '''Sample a chain of multi-hop neighbors. The sign of `this_triple` is to 
            make its `triple_structure`, e.g. ['h_e', 'r_e', 't_e'], which should
            not be an integer.'''
        multi_hop_neighbor = []
        ents_get_nb = [['h_%s'%this_triple_sign, this_triple['head_entity']], 
                    ['t_%s'%this_triple_sign, this_triple['tail_entity']['value']]]
        if not allow_ent_repeat:
            sampled_ents = set([this_triple['head_entity']['id']])
            if self.is_item_value_with_id(this_triple['tail_entity']['value']):
                sampled_ents.add(this_triple['tail_entity']['value']['id'])
        while len(ents_get_nb) > 0:
            rng.shuffle(ents_get_nb)
            now_ent_mark = None
            for i in range(len(ents_get_nb)):
                if self.is_item_value_with_id(ents_get_nb[i][1]):
                    now_ent_mark, now_ent = ents_get_nb[i]
                    ents_get_nb = ents_get_nb[i+1:]
                    break
            if now_ent_mark == None:
                break
            for i in [0, 1] if rng.binomial(1, 0.5) == 0 else [1, 0]:
                if i == 0: # now as head
                    triple = self.sample_triple_given_head(now_ent, 
                        ent_inside_filtered, property_inside_filtered, rng = rng)
                    if triple == None:
                        continue # not found triple
                    else:
                        if not allow_ent_repeat and self.is_item_value_with_id(triple['tail_entity']['value']):
                            if triple['tail_entity']['value']['id'] in sampled_ents:
                                continue # repeat entity
                            else:
                                sampled_ents.add(triple['tail_entity']['value']['id'])
                        new_ent_mark = 'e_%s'%len(multi_hop_neighbor)
                        new_prop_mark = 'r_%s'%len(multi_hop_neighbor)
                        gen_neighbor = {
                            'triple_structure': [now_ent_mark, new_prop_mark, new_ent_mark],
                            'triple': triple
                        } 
                        multi_hop_neighbor.append(gen_neighbor)
                        ents_get_nb.append([new_ent_mark, triple['tail_entity']['value']])
                        break
                else: # now as tail
                    triple = self.sample_triple_given_tail(now_ent, 
                        ent_inside_filtered, property_inside_filtered, rng = rng)
                    if triple == None:
                        continue # not found triple
                    else:
                        if not allow_ent_repeat:
                            if triple['head_entity']['id'] in sampled_ents:
                                continue # repeat entity
                            else:
                                sampled_ents.add(triple['head_entity']['id'])
                        new_ent_mark = 'e_%s'%len(multi_hop_neighbor)
                        new_prop_mark = 'r_%s'%len(multi_hop_neighbor)    
                        gen_neighbor = {
                            'triple_structure': [new_ent_mark, new_prop_mark, now_ent_mark],
                            'triple': triple
                        }
                        multi_hop_neighbor.append(gen_neighbor)
                        ents_get_nb.append([new_ent_mark, gen_neighbor['triple']['head_entity']])
                        break
            if len(multi_hop_neighbor) >= neighbor_n:
                break
        return multi_hop_neighbor

    def has_other_heads(self, triple:str, max_search = 8)->bool:
        ''' Determine whether has other heads except the head in the triple, 
        point to the tail in the triple with the property in the triple. 
        Can be uesed to judge whether the head entity can be uniquely determined 
        by the inversed property and the tail entity.
        '''
        if not self.is_item_value_with_id(triple['tail_entity']['value']):
            # Assume that the tails which is not entity would have multiple heads.
            return True
        prop_id = triple['property']['id']
        tail_id = triple['tail_entity']['value']['id']
        query_dict = {'claims': [prop_id, tail_id]}
        total_count, max_score, scores, results = self.esfwe.es_multi_fields_docs_match(
            query_dict, max_search, index = 'all_entity', add_tqdm=False)
        if total_count > max_search:
            return True
        head_id = triple['head_entity']['id']
        for r in results:
            clms = eval(r['claims'])
            if r['id'] == head_id or prop_id not in clms.keys():
                continue
            for v in clms[prop_id]:
                if v['mainsnak']['snaktype'] != 'value' or v['mainsnak']['datatype'] != 'wikibase-item':
                    continue
                if v['mainsnak']['datavalue']['value']['id'] == tail_id:
                    return True
        return False
     
    def sample_neighbor_path(self, this_triple:dict, this_triple_sign = 'e', 
                neighbor_n = 3, double_path_prior_p = 0.1, rng:RNG = None):
        '''This function sample a path of triples to predict an entity, or a pair of 
        opposite direction paths which share a same entity prediction.
        `double_path_prior_p`: (0, 1), the priority sampling probability of double 
            paths used for reasoning samples.
        return: {
            'path_type': 'single',
            'simple_paths': [['e_0', 'frs_0', 'brm_e', ...], [if path_type is double]],
            'complete_paths': [
                [['e_0', 'r_1', 't_e', 'f', 's'], ['t_e', 'r_e', 'h_e', 'b', 'm'], ...], 
                [[],[], ... if path_type is double]
            ],
            'neighbor_triples': {'0': neighbor_triple_0, '2': neighbor_triple_2, ...}
        } 
        `frs_*` means forward property that has single value. 
        `brm_*` means backward property that has multiple value. 
        `frm_*` and `brs_*` are defined accordingly.
        NOTE that only the property at the end of the path can have multiple value.'''
        # Sample neighbers
        all_triples = {this_triple_sign: {'triple_structure': ['h_%s'%this_triple_sign, 
            'r_%s'%this_triple_sign, 't_%s'%this_triple_sign], 'triple': this_triple}}
        if neighbor_n > 0:
            neighbors = self.sample_multi_hop_neighbor(this_triple, this_triple_sign, neighbor_n, rng = rng)
            for i, nb in enumerate(neighbors):
                all_triples[str(i)] = nb
        # Structure node mapping
        for v in all_triples.values():
            v['triple']['single_inverse'] = not self.has_other_heads(v['triple'])
        node_map = {}
        for v in all_triples.values():
            (h, r, t), tp = v['triple_structure'], v['triple']
            if h not in node_map.keys(): node_map[h] = []
            if t not in node_map.keys(): node_map[t] = []
            forward_single_value = 's' if tp['property']['single_value'] else 'm'
            backward_single_value = 's' if tp['single_inverse'] else 'm'
            node_map[h].append(['out', r, t, forward_single_value, backward_single_value])
            node_map[t].append(['in', r, h, forward_single_value, backward_single_value])
        # Set sampling priority for nodes selected as path end
        classed_nodes = [[]]
        for n in node_map.keys():
            if len(node_map[n]) == 1: classed_nodes.append(n) # end point
            else: classed_nodes[0].append(n) # middle point
        single_sample_p = (1 - double_path_prior_p)/2
        classed_nodes_order = rng.choice([0,1,2], 3, False, [double_path_prior_p, 
            single_sample_p, single_sample_p], shuffle = False)
        node_list = []
        for i in classed_nodes_order:
            if i == 0: 
                rng.shuffle(classed_nodes[0])
                node_list.extend(classed_nodes[0])
            else: 
                node_list.append(classed_nodes[i])
        # Sample generality paths
        def extend_path(path:List):
            '''Extend path along a direction; where len(path) >= 1; path[*] = [e_*, 
                r_*, e_*, 's' or 'm', 'f' or 'b']. NOTE that the path order is 
                reversed since using append.'''
            h, _, t, _, _ = path[-1]
            for in_out, new_r, new_h, fwdsv, bwdsv in node_map[h]: # ['in', 'r_e', 'h_e', 's', 'm']
                if new_h != t:
                    single_value = fwdsv if in_out == 'in' else bwdsv
                    if single_value == 'm':
                        return path
                    fwd_or_bwd = 'f' if in_out == 'in' else 'b'
                    path.append([new_h, new_r, h, fwd_or_bwd, single_value])
                    return extend_path(path)
            return path
        def valid_paths(paths:List[List]): # whether include edit property
            for path in paths:
                for p in path: 
                    if p[1] == 'r_%s'%this_triple_sign: 
                        return True
            return False
        for end_node in node_list:
            paths = []
            for mp in node_map[end_node]:
                in_out, new_r, new_h, fwdsv, bwdsv = mp
                single_value = fwdsv if in_out == 'in' else bwdsv
                fwd_or_bwd = 'f' if in_out == 'in' else 'b'
                path = [[new_h, new_r, end_node, fwd_or_bwd, single_value]]
                paths.append(extend_path(path))
            if valid_paths(paths):
                for path in paths:
                    path.reverse()
                break
        # process return 
        neighbor_triples = {}
        simple_paths = []
        for path in paths:
            simple_paths.append([path[0][0]])
            for t, r, h, s, f in path:
                simple_paths[-1].append('%sr%s_%s'%(s, f, r.split('_')[-1]))
                for e in [t, r, h]:
                    i = e.split('_')[-1]
                    if i not in neighbor_triples.keys():
                        neighbor_triples[i] = all_triples[i]
        neighbor_triples.pop(this_triple_sign)
        return_res = {
            'path_type': 'single' if len(paths) == 1 else 'double',
            'simple_paths': simple_paths,
            'complete_paths': paths,
            'neighbor_triples': neighbor_triples
        }
        return return_res
    
    def sample_locality_path(self, edit_triple:Dict, gen_neighbor_triples:List[Dict], 
            neighbor_n = 4, double_path_prior_p = 0.1, max_sample_try = 3, 
            es_match_size = 1024, ent_inside_filtered = True, 
            property_inside_filtered = True, rng:RNG = None):
        '''`gen_neighbor_triples`: neighbor triples to avoid conflict between 
        locality and generality paths.'''
        def get_triple_hashes(triple:Dict):
            # Construct triple hashes to avoid Locality triples cover on Generality triples.
            head_id = 'ID:%s'%triple['head_entity']['id']
            if triple['tail_entity']['datatype'] == 'wikibase-item':
                # Overlook property
                tail_id = 'ID:%s'%triple['tail_entity']['value']['id']
                return [(head_id, tail_id), (tail_id, head_id)]
            else:
                prop_id = 'ID:%s'%triple['property']['id'] 
                tail_str = '%s:%s'%(triple['tail_entity']['datatype'], 
                                    str(triple['tail_entity']['value']))
                return [(head_id, prop_id, tail_str)]
        triple_hash = set(get_triple_hashes(edit_triple))
        for t in gen_neighbor_triples:
            triple_hash.union(get_triple_hashes(t))
        # Sampling Locality samples.
        loc_types = ['h', 'r', 'n']
        if edit_triple['tail_entity']['datatype'] == "wikibase-item":
            loc_types.append('t')
        rng.shuffle(loc_types)
        for lt in loc_types:
            if lt == 'r': # r
                ent = edit_triple['property']
            else: # h/n/t
                if lt == 'n':  # n: none of component share with edit
                    ent = {'id': self.random_filtered_entid(rng)}
                else: # h/t: related to edit head or tail
                    ent = edit_triple['head_entity'] if lt == 'h' else edit_triple['tail_entity']['value']
            matched_ents_cache = None
            for _ in range(max_sample_try):
                # Sample center of locality path
                if lt == 'r': # r: related to edit property
                    loc_triple, matched_ents_cache = self.sample_triple_given_property(
                        ent, ent_inside_filtered, ent_inside_filtered, 
                        es_match_size, matched_ents_cache, True, rng = rng)
                    if loc_triple == None: break
                else: # h/t/n
                    for j in [0, 1] if rng.binomial(1, 0.5) else [1, 0]:
                        if j == 0: # used as head
                            loc_triple = self.sample_triple_given_head(ent, 
                                ent_inside_filtered, property_inside_filtered, rng = rng)
                            if loc_triple != None: break
                        else: # used as tail
                            loc_triple, matched_ents_cache = self.sample_triple_given_tail(
                                ent, ent_inside_filtered, property_inside_filtered, 
                                es_match_size, matched_ents_cache = matched_ents_cache, 
                                return_matched_ents = True, rng = rng)
                            if loc_triple != None: break
                    if loc_triple == None: break
                if any(hs in triple_hash for hs in get_triple_hashes(loc_triple)): continue
                # Sample path and check whether it does not have triples in Generality domain
                loc_path = self.sample_neighbor_path(loc_triple, 'l', neighbor_n, double_path_prior_p, rng = rng) 
                flg = False
                for t in loc_path['neighbor_triples'].values():
                    if any(hs in triple_hash for hs in get_triple_hashes(t['triple'])):
                        flg = True
                        break
                if flg: continue
                # Create path
                loc_path = {
                    'loc_type': {'h': 'head of edit', 'r': 'property of edit', 
                        't': 'tail of edit', 'n': 'none'}[lt],
                    'path_type': loc_path['path_type'],
                    'simple_paths': loc_path['simple_paths'],
                    'complete_paths': loc_path['complete_paths'],
                    'loc_triples': loc_path['neighbor_triples']
                }
                loc_path['loc_triples']['l'] = {
                    'triple_structure': ['h_l', 'r_l', 't_l'],
                    'triple': loc_triple
                }
                return loc_path
        return None 

    def sample_a_structured_data_given_edit_triple(self, edit_triple:Dict, 
            max_hop = 4, gen_n = 1, loc_n = 1, es_match_size = 1024, 
            double_path_prior_p = 0.05, rng:RNG = None, max_resample = 3):  
        '''Sample a structure data, including edit, generality, and locality samples.'''
        # get Generality
        gen_ds = {}
        for neighbor_n in rng.choice(list(range(max_hop)), gen_n * max_resample, True):
            gen_path = self.sample_neighbor_path(edit_triple, 'e', neighbor_n, double_path_prior_p, rng = rng)
            if any([gen_path['simple_paths'] == gd['simple_paths'] for gd in gen_ds.values()]):
                # Drop repeated generality path
                continue
            gen_ds[str(len(gen_ds))] = gen_path
            if len(gen_ds) >= gen_n: break
        if len(gen_ds) < gen_n - 1: raise
        # get Locality
        loc_ds = {}
        gen_neighbor_triples = [g['triple'] for gen_d in gen_ds.values() for g in gen_d['neighbor_triples'].values()]
        for neighbor_n in rng.choice(list(range(max_hop)), loc_n * max_resample, True):
            loc_path = self.sample_locality_path(edit_triple, 
                gen_neighbor_triples, neighbor_n, double_path_prior_p, 
                3, es_match_size, True, True, rng = rng)
            if loc_path != None:
                if any([loc_path['simple_paths'] == ld['simple_paths'] for ld in loc_ds.values()]):
                    # Drop repeated locality path
                    continue
                loc_ds[str(len(loc_ds))] = loc_path
            if len(loc_ds) >= loc_n: break
        if len(loc_ds) < loc_n - 1: raise
        return_d = {
            'edit': edit_triple,
            'generality': gen_ds,
            'locality': loc_ds
        }
        self.get_details_for_ents_with_id(return_d, False) 
        return return_d
