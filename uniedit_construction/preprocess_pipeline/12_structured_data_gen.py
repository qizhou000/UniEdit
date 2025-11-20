#%% 
from uniedit_construction.tools.edit_data_structure_sample import EditDataStructureSampler 
from decimal import Decimal
import os, argparse, sys
from tqdm import tqdm
from time import time
import numpy as np
import json
 
def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--subject', type=str, required = True)
    parser.add_argument('-bs', '--batch_size', type=int, default = 1000)
    parser.add_argument('-mh', '--max_hop', type=int, default = 4)
    parser.add_argument('-gn', '--gen_n', type=int, default = 1)
    parser.add_argument('-ln', '--loc_n', type=int, default = 1)
    parser.add_argument('-ems', '--es_match_size', type=int, default = 1024)
    parser.add_argument('-dppp', '--double_path_prior_p', type=float, default = 0.05)
    args = parser.parse_args()
    return args
def decimal_to_float(x):
    return float(x) if isinstance(x, Decimal) else x
 
class cfg:
    subject = 'biology'
    batch_size = 1000
    max_hop = 3
    gen_n = 2
    loc_n = 2
    es_match_size = 1024
    double_path_prior_p = 0.05
cfg = get_attr()
rng = np.random.default_rng(1234)
with open(os.path.join('data/wikidata/s11_edit_triples_post_proc', '%s.json'%cfg.subject), 'r') as f:
    edit_triples = json.load(f)
edss = EditDataStructureSampler(cfg.subject) # biology_debug
structured_data = {}
save_dir = os.path.join('data/wikidata/s12_structured_data', cfg.subject)
if not os.path.exists(save_dir): 
    os.makedirs(save_dir)
edit_triple_index = [str(i) for i in sorted([int(i) for i in edit_triples.keys()])]
now_proc_n = 0
for i in tqdm(edit_triple_index, 'Data generating'):
    try:
        structured_data[i] = edss.sample_a_structured_data_given_edit_triple(
            edit_triples[i]['edit'], cfg.max_hop, cfg.gen_n, cfg.loc_n, cfg.es_match_size, 
            cfg.double_path_prior_p, rng)
    except: 
        print('Error: %s, skip.'%i)
        continue
    if len(structured_data) % cfg.batch_size == 0 and len(structured_data) > 0:
        now_proc_n += len(structured_data)
        save_path = os.path.join(save_dir, '%s.json'%now_proc_n)
        with open(save_path, 'w') as f:
            json.dump(structured_data, f, default = decimal_to_float)
        structured_data = {}
if len(structured_data) > 0:
    now_proc_n += len(structured_data)
    save_path = os.path.join(save_dir, '%s.json'%now_proc_n)
    with open(save_path, 'w') as f:
        json.dump(structured_data, f, default = decimal_to_float)

