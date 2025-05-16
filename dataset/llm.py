#%%
from typing import Dict, List, Tuple, Union
from transformers import AutoTokenizer
import torch, os, json, re
from abc import ABC, abstractmethod
from . import BaseEditData
import numpy as np
from tqdm import tqdm
from copy import deepcopy

    
class BaseLLMEditData(BaseEditData):
    '''
    Functions used to read and preprocess various LLM editing datasets, which 
    structures a dataset as a list like [
        { # test1
            'requests': [
                {'prompt': str, 'target_new': str, 'subject': str},
                {'prompt': str, 'target_new': str, 'subject': str}, ...
            ],
            'generality': {
                'gen_1_name':[
                    {'prompt': str, 'target': str, "related_hops": [# Optional, for multi-hops
                        {'prompt': str, 'target': str, ...},
                        {'prompt': str, 'target': str, ...}, ...], 
                     'target_check_topk': int, # Optional, for evaluation
                     ...},
                    {'prompt': str, 'target': str, "related_hops": [For multi-hops], ...}, ...
                ],
                'gen_2_name':[...], ...
            },
            'locality': {
                'loc_1_name':[
                    {'prompt': str, 'target': str, ...}, ...
                ],
                'loc_2_name':[...], ...
            }
        }, 
        { # test2
            'requests': ...
        }, ...
    ]. 
    '''
    def __init__(self, random_seed) -> None:
        super().__init__() 
        self.rng = np.random.default_rng(random_seed)
        # check up data structure: requests/generality/locality
        d = self.get_data_by_ids([0])[0]
        assert 'prompt' in d['requests'][0].keys() # requests
        assert 'target_new' in d['requests'][0].keys()
        if len(d['generality'].keys()) > 0:
            gen_name = list(d['generality'].keys())[0] # generality
            assert 'prompt' in d['generality'][gen_name][0].keys()
            assert 'target' in d['generality'][gen_name][0].keys()
        if len(d['locality'].keys()) > 0:
            loc_name = list(d['locality'].keys())[0] # locality
            assert 'prompt' in d['locality'][loc_name][0].keys()
            assert 'target' in d['locality'][loc_name][0].keys()

    def __load_splited_wiki__(self, data_dir = None, wiki_sentence_truncation = 128):
        if data_dir == None:
            data_dir = 'data/meta-train/comprehensive/wikitext/splited_sentences'
        assert os.path.isdir(data_dir) 
        self.wiki_sentences = []
        for d_name in tqdm(os.listdir(data_dir), 'Preparing splited wiki...', ncols = 60):
            if '.json' not in d_name:
                continue
            data_path = os.path.join(data_dir, d_name)
            with open(data_path, 'r') as f:
                self.wiki_sentences.extend(json.load(f))
        self.wiki_sentence_n = len(self.wiki_sentences)
        self.wiki_sentence_truncation = wiki_sentence_truncation
    
    def __add_wiki_loc_data__(self, a_data):
        stc = self.wiki_sentences[self.rng.integers(0, self.wiki_sentence_n)]
        wds = stc.split(' ')
        for i in range(len(wds)):
            if wds[i] != '':
                break
        prmpt = ' '.join(wds[:i + 1])
        tgt = ' '.join(wds[i + 1:self.wiki_sentence_truncation])
        a_data['locality']['wiki_loc'] = [{'prompt': prmpt, 'target': tgt}]

class ZSRE(BaseLLMEditData):
    def __init__(self, data_path:str = 'data/meta-train/zsre/zsre_mend_train.json', 
            data_n = None, random_seed = 1234, add_wiki_loc = True, 
            wiki_data_path = None, wiki_sentence_truncation = 128):
        # load and pre-process ZSRE dataset
        with open(data_path, 'r') as f: 
            data = json.load(f)
        self.sample_count = min(len(data), data_n) if data_n != None else len(data)
        self.prepare_data = []
        prefix = ' The answer is: '
        for d in tqdm(data[:self.sample_count], 'ZSRE data preparing...', ncols = 60):
            self.prepare_data.append({ 
                'requests': [{'prompt': d['src'] + prefix, 'target_new': d['alt'], 
                    'subject': d['subject'], 'ground_truth': d['answers'][0]
                }],
                'generality': {
                    'rephrase':[{'prompt': d['rephrase'] + prefix, 'target': d['alt'], 'subject': d['subject']}],
                },
                'locality': {'NQ':[{'prompt': d['loc'], 'target': d['loc_ans']}],
                }
            })
        # for wiki locality 
        self.add_wiki_loc = add_wiki_loc
        if add_wiki_loc:
            self.__load_splited_wiki__(wiki_data_path, wiki_sentence_truncation)  
        super().__init__(random_seed)
 
    def get_data_by_ids(self, ids:List[int]):
        return_data = [self.prepare_data[i] for i in ids]
        if self.add_wiki_loc:
            for d in return_data:
                self.__add_wiki_loc_data__(d)
        return return_data

    def data_n(self): 
        return self.sample_count

    def dataset_name(self):
        return 'ZSRE'


class Counterfact(BaseLLMEditData):
    def __init__(self, data_path:str = 'data/meta-train/cf/counterfact-train.json', 
            data_n = None, random_seed = 1234, add_wiki_loc = True, 
            wiki_data_path = None, wiki_sentence_truncation = 128):
        # load and pre-process Counterfact dataset
        with open(data_path, 'r') as f: 
            data = json.load(f)
        self.sample_count = min(len(data), data_n) if data_n != None else len(data)
        self.prepare_data = []
        for d in tqdm(data[:self.sample_count], 'Counterfact data preparing...', ncols = 60):
            self.prepare_data.append({ 
                'requests': [{'prompt': d['prompt'], 'target_new': d['target_new'], 
                    'subject': d['subject'], 'ground_truth': d['ground_truth']
                }],
                'generality': {
                    'rephrase':[{'prompt': d['rephrase_prompt'], 'target': d['target_new'], 'subject': d['subject']}],
                },
                'locality': {'NQ':[{'prompt': d['locality_prompt'], 'target': d['locality_ground_truth']}],
                }
            })
        # for wiki locality 
        self.add_wiki_loc = add_wiki_loc
        if add_wiki_loc:
            self.__load_splited_wiki__(wiki_data_path, wiki_sentence_truncation)  
        super().__init__(random_seed)
 
    def get_data_by_ids(self, ids:List[int]):
        return_data = [self.prepare_data[i] for i in ids]
        if self.add_wiki_loc:
            for d in return_data:
                self.__add_wiki_loc_data__(d)
        return return_data

    def data_n(self): 
        return self.sample_count
    
    def dataset_name(self):
        return 'Counterfact'


class RippleEffect(BaseLLMEditData):
    def __init__(self, data_path:str = 'data/meta-train/ripple_effect/ripe_train.json', 
            data_n = None, random_seed = 1234, add_wiki_loc = True, 
            wiki_data_path = None, wiki_sentence_truncation = 128):
        with open(data_path, 'r') as f: 
            data = json.load(f)
        self.sample_count = min(len(data), data_n) if data_n != None else len(data)
        self.prepare_data = []
        data_types = {
            'generality': ['Logical_Generalization', 'Compositionality_I', 
                            'Compositionality_II', 'Subject_Aliasing'],
            'locality': ['Relation_Specificity', 'Forgetfulness']
        }
        for d in tqdm(data[:self.sample_count], 'Ripple Effect data preparing...', ncols = 60):
            pd = {
                'requests': [{
                    'prompt': d['prompt'], 'target_new': d['target_new'], 
                    'subject': d['subject']
                }],
                'generality':{},
                'locality':{}
            }
            for gen_loc_type in data_types:
                for data_type in data_types[gen_loc_type]:
                    for type_data in d[data_type]:
                        if type_data['prompt'] != '' and len(type_data['targets']) > 0 and type_data['targets'][0] != '':
                            if data_type not in pd[gen_loc_type]:
                                pd[gen_loc_type][data_type] = []
                            pd[gen_loc_type][data_type].append({
                                'prompt': type_data['prompt'], 
                                'target': type_data['targets'][0]
                            })
                            if gen_loc_type == 'generality':
                                pd[gen_loc_type][data_type][-1]['subject'] = d['subject']
            self.prepare_data.append(pd)
        # for wiki locality 
        self.add_wiki_loc = add_wiki_loc
        if add_wiki_loc:
            self.__load_splited_wiki__(wiki_data_path, wiki_sentence_truncation)  
        super().__init__(random_seed)
 
    def get_data_by_ids(self, ids:List[int]):
        return_data = [self.prepare_data[i] for i in ids]
        if self.add_wiki_loc:
            for d in return_data:
                self.__add_wiki_loc_data__(d)
        return return_data

    def data_n(self): 
        return self.sample_count

    def dataset_name(self):
        return 'RippleEffect'




class UniEdit(BaseLLMEditData):
    def __init__(self, data_dir:str = 'data/UniEdit/test', # or 'data/UniEdit/test'
            data_n = None, disciplines: List[str] = [], 
            has_gen_patterns:List[str] = None, has_loc_patterns:List[str] = None, 
            random_seed = 1234, add_wiki_loc = False, wiki_data_path = None, 
            wiki_sentence_truncation = 128):
        '''Data structure: [
            {
                "index": "biology-1234",
                "requests": [{"prompt": str, "target_new": str, "subject": str}],
                "generality": {
                    "uni_gen": [
                        {
                            "prompt": str, "target": str, "related_hops": [
                                {"prompt": str, "target": str, "subject": str},
                                {"prompt": str, "target": str, "subject": str}, ...],
                            "patterns": {"rephrase": bool, ...}
                        }, ...
                    ]
                },
                "locality": {
                    "uni_loc":[
                        {
                            "prompt": str, "target": str, 
                            "related_hops": [
                                {"prompt": str, "target": str, "subject": str},
                                {"prompt": str, "target": str, "subject": str}, ...],
                            "patterns": {"subject specificity": bool, ...}
                        }, ...
                    ]
                }
            }, ...
        ]'''
        def extract_pts(one_hop):
            res = {"prompt": one_hop['prompt'], "target": one_hop['target']}
            if 'subject' in one_hop: res['subject'] = one_hop['subject']
            return res
        self.specified_disciplines = []
        self.original_data = {}
        self.prepare_data = [] 
        for data_name in tqdm(os.listdir(data_dir), "Loading UniEdit from %s"%data_dir, ncols = 60):
            discipline_name = data_name.split('.')[0]
            if disciplines and discipline_name not in disciplines:
                continue
            if discipline_name in disciplines:
                self.specified_disciplines.append(discipline_name)
            print(f'Loading: {discipline_name}.')
            data_path = os.path.join(data_dir, data_name)
            with open(data_path, 'r') as f:
                data = json.load(f)
            for i, d in data.items():
                has_pattern_flag = False
                if has_gen_patterns == None and has_loc_patterns == None:
                    has_pattern_flag = True
                new_d = {
                    'index': '%s-%s'%(discipline_name, i),
                    'requests': [{"prompt": d['edit']['prompt'], 
                        "target_new": d['edit']['target'], 
                        "subject": d['edit']['subject']}], 
                    'generality': {"uni_gen": []}, 'locality': {"uni_loc": []}
                }
                for gl in ['generality', 'locality']:
                    for gld in d[gl].values():
                        extracted_gld = extract_pts(gld)
                        one_hops = []
                        if gld['path_type'] == 'single':
                            one_hops.extend(gld['one_hops'])
                        else:
                            one_hops.extend(gld['single_path1']['one_hops'])
                            one_hops.extend(gld['single_path2']['one_hops'])
                        extracted_gld['related_hops'] = [extract_pts(oh) for oh 
                            in one_hops if oh['path'] != ["h_e", "r_e", "t_e"]]
                        if gl == 'generality': 
                            patterns = self.get_gen_patterns(gld)
                            extracted_gld['patterns'] = patterns
                            new_d[gl]['uni_gen'].append(extracted_gld)
                            if not has_pattern_flag and has_gen_patterns != None:
                                for p in has_gen_patterns:
                                    if patterns[p]:
                                        has_pattern_flag = True
                                        break
                        else:
                            patterns = self.get_loc_patterns(d['edit'], gld)
                            extracted_gld['patterns'] = patterns
                            new_d[gl]['uni_loc'].append(extracted_gld)
                            if not has_pattern_flag and has_loc_patterns != None:
                                for p in has_loc_patterns:
                                    if patterns[p]:
                                        has_pattern_flag = True
                                        break
                        if patterns['same entity reason']:
                            extracted_gld['prompt'] = extracted_gld['prompt'] + ' The answer is (Yes/No):'
                            extracted_gld['target_check_topk'] = 1
                if has_pattern_flag:
                    self.prepare_data.append(new_d)
                    self.original_data[new_d['index']] = d
        assert len(self.prepare_data) > 0
        rng = np.random.default_rng(random_seed)
        rng.shuffle(self.prepare_data)
        self.sample_count = min(len(self.prepare_data), data_n) if data_n != None else len(self.prepare_data)
        self.prepare_data = self.prepare_data[:self.sample_count]
        # for wiki locality 
        self.add_wiki_loc = add_wiki_loc
        if add_wiki_loc:
            self.__load_splited_wiki__(wiki_data_path, wiki_sentence_truncation)  
        super().__init__(random_seed)
        
    def get_data_by_ids(self, ids:List[int]):
        return_data = [self.prepare_data[i] for i in ids]
        if self.add_wiki_loc:
            for d in return_data:
                self.__add_wiki_loc_data__(d)
        return return_data

    def get_data_by_ids_without_loc_edit_collision(self, ids:List[int]):
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
        return_data = deepcopy(self.get_data_by_ids(ids))
        edit_hashes = set()
        for rd in return_data:
            edit_hashes = edit_hashes.union(get_triple_hashes(self.original_data[rd['index']]['edit']))
        def loc_edit_collide(ori_ld):
            one_hops = []
            if ori_ld['path_type'] == 'single':
                one_hops.extend(ori_ld['one_hops'])
            else:
                one_hops.extend(ori_ld['single_path1']['one_hops'])
                one_hops.extend(ori_ld['single_path2']['one_hops'])
            for oh in one_hops:
                for hs in get_triple_hashes(oh):
                    if hs in edit_hashes:
                        return True
            return False
        for rd in return_data:
            assert len(self.original_data[rd['index']]['locality']) == 1
            assert len(rd['locality']['uni_loc']) == 1
            ori_ld = self.original_data[rd['index']]['locality']['0']
            if loc_edit_collide(ori_ld):
                rd['locality']['uni_loc'] = []
                print('Find a locality collision & remove: ', rd['index'])
        return return_data

    def data_n(self): 
        return self.sample_count

    def dataset_name(self):
        if len(self.specified_disciplines) > 0:
            return 'UniEdit-' + '-'.join(self.specified_disciplines)
        return 'UniEdit'

    def get_gen_patterns(self, gd):
        def is_rep():
            if gd['path_type'] != 'single':
                return False
            paths = gd['paths']
            if len(paths) != 1:
                return False
            path = paths[0]
            if len(path) != 5:
                raise
            if path[:4] != ['h_e', 'r_e', 't_e', 'f']:
                return False
            if path[4] not in ('s', 'm'):
                raise
            return True
        def is_mh():
            if gd['path_type'] == 'single':
                paths = gd['paths']
                return len(paths) >= 2
            else :
                paths1 = gd['single_path1']['paths']
                paths2 = gd['single_path2']['paths']
                return len(paths1) >= 2 or len(paths2) >= 2
        def is_rr():
            paths = []
            if gd['path_type'] == 'single':
                paths.extend(gd['paths'])
            else:
                paths.extend(gd['single_path1']['paths'])
                paths.extend(gd['single_path2']['paths'])
            for path in paths:
                if len(path) != 5: raise
                if path[:4] == ['t_e', 'r_e', 'h_e', 'b']:
                    return True
            return False
        def is_ser():
            return gd['path_type'] == 'double'
        def is_sa():
            one_hops = []
            if gd['path_type'] == 'single':
                one_hops.extend(gd['one_hops'])
            else:
                one_hops.extend(gd['single_path1']['one_hops'])
                one_hops.extend(gd['single_path2']['one_hops'])
            assert ['h_e', 'r_e', 't_e'] in [h['path'] for h in one_hops]
            for hop in one_hops:
                if hop['path'] == ['h_e', 'r_e', 't_e']:
                    head = hop['head_entity']
                    if head['aliases'] != [] and hop['subject'] != head['label']:
                        return True
            return False
        def is_oa():
            one_hops = []
            if gd['path_type'] == 'single':
                one_hops.extend(gd['one_hops'])
            else:
                one_hops.extend(gd['single_path1']['one_hops'])
                one_hops.extend(gd['single_path2']['one_hops'])
            for hop in one_hops:
                if hop['path'] == ['h_e', 'r_e', 't_e']:
                    if hop['tail_entity']['datatype'] != 'wikibase-item':
                        return False
                    if hop['tail_entity']['value']['aliases'] == []:
                        return False
                    if hop['target'] == hop['tail_entity']['value']['label']:
                        return False
                    return True
        return {'rephrase': is_rep(), 'multi-hop':is_mh(), 'relation reverse': is_rr(),
            'same entity reason': is_ser(), 'subject alias': is_sa(), 'object alias': is_oa()}

    def get_loc_patterns(self, ed, ld):
        def is_mh():
            if ld['path_type'] == 'single':
                paths = ld['paths']
                return len(paths) >= 2
            else :
                paths1 = ld['single_path1']['paths']
                paths2 = ld['single_path2']['paths']
                return len(paths1) >= 2 or len(paths2) >= 2
        def is_ser():
            return ld['path_type'] == 'double'
        def is_ss():
            head_id = ed['head_entity']['id']
            path_type = ld['path_type']
            one_hops = []
            if path_type == 'single' :
                one_hops.extend(ld['one_hops'])
            else:
                one_hops.extend(ld['single_path1']['one_hops'])
                one_hops.extend(ld['single_path2']['one_hops'])
            for hop in one_hops:
                if hop['head_entity']['id'] == head_id:
                    return True
                if hop['tail_entity']['datatype'] == 'wikibase-item' and hop['tail_entity']['value']['id'] == head_id:
                    return True
            return False
        def is_rs():
            prop_id = ed['property']['id']
            path_type = ld['path_type']
            one_hops = []
            if path_type == 'single':
                one_hops.extend(ld['one_hops'])
            else:
                one_hops.extend(ld['single_path1']['one_hops'])
                one_hops.extend(ld['single_path2']['one_hops'])
            for hop in one_hops:
                if hop['property']['id'] == prop_id:
                    return True
            return False
        def is_os():
            if ed['tail_entity']['datatype'] != 'wikibase-item':
                return False
            target_tail_id = ed['tail_entity']['value']['id']
            path_type = ld['path_type']
            one_hops = []
            if path_type == 'single' :
                one_hops.extend(ld['one_hops'])
            else:
                one_hops.extend(ld['single_path1']['one_hops'])
                one_hops.extend(ld['single_path2']['one_hops'])
            for hop in one_hops:
                if hop['head_entity']['id'] == target_tail_id:
                    return True
                if hop['tail_entity']['datatype'] == 'wikibase-item' and hop['tail_entity']['value']['id'] == target_tail_id:
                    return True
            return False
        def is_1_NF():
            head_id = ed['head_entity']['id']
            target_pid = ed['property']['id']
            path_type = ld['path_type']
            one_hops = []
            if path_type == 'single':
                one_hops.extend(ld['one_hops'])
            else:
                one_hops.extend(ld['single_path1']['one_hops'])
                one_hops.extend(ld['single_path2']['one_hops'])
            for hop in one_hops:
                if hop['property']['id'] == target_pid and hop['head_entity']['id'] == head_id:
                    return True
            return False
        return {'multi-hop': is_mh(), 'same entity reason': is_ser(), 
            'subject specificity': is_ss(), 'relation specificity': is_rs(),
            'object specificity': is_os(), '1-N forgotten': is_1_NF()}

