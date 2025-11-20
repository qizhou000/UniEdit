#%% Control parts of properties that not particularly relevant to subject knowledge 
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from time import time, sleep
from collections import Counter
from decimal import Decimal
from typing import Dict, List
import os, argparse, sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

def get_data_count(data:Dict):
    n = 0
    for sub in data.keys():
        print(sub, len(data[sub]))
        n += len(data[sub])
    print('Total number of data:', n)
def get_property_count(data:Dict, top_k = 10):
    propers_count = {'all': {}}
    for sub in data.keys():
        propers_count[sub] = {}
        for d in data[sub]:
            prop = d['property']['label']
            if prop not in propers_count[sub]:
                propers_count[sub][prop] = 0
            if prop not in propers_count['all']:
                propers_count['all'][prop] = 0
            propers_count[sub][prop] += 1
            propers_count['all'][prop] += 1
    for sub in propers_count.keys():
        print(sub)
        print(sorted(propers_count[sub].items(), key=lambda x: x[1], reverse=True)[:top_k])
        print()
def get_head_description_word_count(data:Dict, ngram = 2, top_k = 10):
    all_sub_grams = []
    for sub in data.keys():
        sub_grams = []
        for d in data[sub]:
            tokens = word_tokenize(d['head_entity']['description'])
            sub_grams.extend(list(ngrams(tokens, ngram)))
        word_freq = Counter(sub_grams)
        print(sub,'description token count:')
        for freq in word_freq.most_common(top_k):
            print(freq)
        print()
        all_sub_grams.extend(sub_grams)
    word_freq = Counter(all_sub_grams)
    print('All subjects description token count:')
    for freq in word_freq.most_common(top_k):
        print(freq)
#%% Load all edit triples
all_edit_triples = {}
data_dir = 'data/wikidata/s10_edit_triples'
for data_name in tqdm(os.listdir(data_dir), 'Loading edit triples'):
    path = os.path.join(data_dir, data_name)
    with open(path, 'r') as f:
        all_edit_triples[data_name.split('.')[0]] = json.load(f)
all_edit_heads = {}
data_dir = 'data/wikidata/s9_edit_heads_sampling'
for data_name in tqdm(os.listdir(data_dir), 'Loading edit heads'):
    path = os.path.join(data_dir, data_name)
    heads = pd.read_csv(path).to_dict(orient = 'records')
    for h in heads:
        h['share_sub_words'] = list(eval(h['share_sub_words']))
    all_edit_heads[data_name.split('.')[0]] = {h['id']:h for h in heads}
#%% Statistics of original edit data
get_property_count(all_edit_triples, top_k = 10)
get_data_count(all_edit_triples)
# get_head_description_word_count(all_edit_triples, 2, top_k = 30)
#%% Property control scale
prop_accept_p = {
    'sex or gender': 0.1, 
    # 'chemical formula': 0.35, 
    # 'mass': 0.5, 
    # 'isomeric SMILES': 0.5, 
    # 'canonical SMILES': 0.5, 
    'coordinate location': 0.5, 
    'publication date': 0.5, 
    'copyright status': 0.3, 
    'named after': 0.5, 
    'author name string': 0.3, 
    'language of work or name': 0.3
}
head_description_part_accept_p = {
    'all': {'us patent': 0.05, 'doctoral thesis': 0.1},
    'mathematics': {'archival photograph': 0, 'album': 0, 'video game': 0, 
        'in the constellation': 0, 'DNA': 0, 'gene': 0, 'protein': 0, 
        'prime Minister': 0, 'Grade II listed building': 0, 'product encoded by': 0, 
        'artist': 0, 'film': 0, 'macromolecular': 0, 'archival photograph': 0, 
        'painting': 0,'song': 0, 'group of bus stops': 0, 'chemical': 0, 
        'musical':0
    },
    'chemistry': {'biologist': 0.4},
    'economics': {'painting by': 0.0},
    'pedagogy': {'painting by': 0.0},
    'civil engineering': {'painting by': 0.0, 'Grade II listed': 0.05},
    'psychology': {'painting by': 0.0, 'episode of': 0},
    'medicine': {'hospital in': 0.6, 'episode of': 0, 'painting by': 0.0},
    'agronomy': {'england, uk': 0.2, 'Grade II listed': 0.05, 'painting by': 0.0},
    'philosophy': {'painting by': 0.0, 'episode of': 0, 'film by': 0, 'film direct': 0},
    'environmental science': {'painting by': 0.0, 'episode of': 0},
    'sports science': {'painting by': 0.0, 'episode of': 0},
    'physics': {'painting by': 0.0, 'episode of': 0},
    'geoscience': {'painting by': 0.0, 'episode of': 0},
    'astronomy': {'painting by': 0.0, 'episode of': 0},
    ### 
    'mechanical engineering': {'painting by': 0.0, 'england, uk': 0.2, 
        'video game': 0, 'episode of': 0, 'protein found': 0, 'gene found': 0, 
        'film by': 0, 'film direct': 0, 'album': 0},
}
head_single_sub_words_accept_p = {
    'mathematics': {"number": 0.1, "function": 0.1, "vector": 0.1, "limit": 0.1, 
        "series": 0.1, "sequence": 0.1, "group": 0.1, "ring": 0.1, "field": 0.1, 
        "solution": 0.1, "root": 0.1, "factor": 0.1, "product": 0.1, "sum": 0.1, 
        "difference": 0.1, "union": 0.1, "magnitude": 0.1, "logical": 0.1, 
        "infinite": 0.1, "abstract": 0.1, "real": 0.1, "complex": 0.1, 
        "continuous": 0.1, "positive": 0.1, "negative": 0.1, "even": 0.1, 
        "odd":0.1
    }    
}
#%% Resample edit data
import re
rng = np.random.default_rng(1234)
def if_accept(triple, sub):
    head_id = triple['head_entity']['id']
    head_description = triple['head_entity']['description'].lower()
    head_sub_words = all_edit_heads[sub][head_id]['share_sub_words']
    head_sub_words_n = all_edit_heads[sub][head_id]['share_sub_words_n']
    # property control
    prop_label = triple['property']['label']
    if prop_label in prop_accept_p.keys():
        if rng.binomial(1, prop_accept_p[prop_label]) != 1:
            return False
    # head description part control
    ks = ['all']
    if sub in head_description_part_accept_p.keys():
        ks.append(sub)
    for k in ks:
        for word, p in head_description_part_accept_p[k].items():
            if re.search(r'\b'+word.lower()+r'\b', head_description):
                if rng.binomial(1, p) != 1:
                    return False
    # head single sub words control
    if sub in head_single_sub_words_accept_p.keys() and head_sub_words_n == 1:
        if head_sub_words[0] in head_single_sub_words_accept_p[sub].keys():
            if rng.binomial(1, head_single_sub_words_accept_p[sub][head_sub_words[0]]) != 1:
                return False
    return True
resampled_data = {}
for sub in tqdm(all_edit_triples.keys()):
    resampled_data[sub] = []
    for d in all_edit_triples[sub]:
        if if_accept(d, sub):
            resampled_data[sub].append(d)
get_property_count(resampled_data, top_k = 30)
get_data_count(resampled_data)
# get_head_description_word_count(resampled_data, 2, top_k = 30)
#%% Save post-processed data
save_dir = os.path.join('data/wikidata/s11_edit_triples_post_proc')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for sub in tqdm(resampled_data.keys()):
    save_path = os.path.join(save_dir, '%s.json'%(sub))
    save_data = {str(i): {'edit': d} for i, d in enumerate(resampled_data[sub])}
    with open(save_path, 'w') as f:
        json.dump(save_data, f)


# %%
