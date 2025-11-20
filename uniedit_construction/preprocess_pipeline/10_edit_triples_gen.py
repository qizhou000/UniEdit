#%% 
from uniedit_construction.tools.edit_data_structure_sample import EditDataStructureSampler 
from time import time, sleep
from decimal import Decimal
import os, argparse, sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--subject', type=str, required=True)
    args = parser.parse_args()
    return args
def decimal_to_float(x):
    return float(x) if isinstance(x, Decimal) else x

class cfg:
    subject = 'biology'
cfg = get_attr()
rng = np.random.default_rng(1234)
edss = EditDataStructureSampler(cfg.subject) # biology_debug
edit_head_path = os.path.join('data/wikidata/s9_edit_heads_sampling', '%s.csv'%cfg.subject)
edit_head_ids = [ent['id'] for ent in pd.read_csv(edit_head_path).iloc]
edit_triples = edss.sample_edit_triples_given_head_ids(edit_head_ids, rng = rng)
save_dir = os.path.join('data/wikidata/s10_edit_triples')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '%s.json'%(cfg.subject))
with open(save_path, 'w') as f:
    json.dump(edit_triples, f, default=decimal_to_float)
