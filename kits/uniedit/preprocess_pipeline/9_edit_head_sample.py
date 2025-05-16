#%% 
from kits.uniedit.tools.edit_data_structure_sample import EditDataStructureSampler 
from decimal import Decimal
import os, argparse, sys
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
import json

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--subject', type=str, required=True)
    parser.add_argument('-gn', '--gen_n', type=int, required = True)
    parser.add_argument('-mll', '--max_label_len', type=int, default = 10)
    args = parser.parse_args()
    return args
class cfg:
    subject = 'math'
    gen_n = 200
    max_label_len = 10
cfg = get_attr()
rng = np.random.default_rng(1234)
edss = EditDataStructureSampler(cfg.subject) # biology_debug
# #%% 1. Selection based solely on sorted sampling weight 
# sampled_ent = []
# for i in np.argsort(edss.edit_entity_sample_weight)[::-1]:
#     if len(edss.sub_ent[i]['labels'].split(' ')) > cfg.max_label_len:
#         continue
#     sampled_ent.append(edss.sub_ent[i])
#     if len(sampled_ent) >= cfg.gen_n:
#         break
# save_dir = os.path.join('data/wikidata/s9_edit_heads_direct_selection')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# save_path = os.path.join(save_dir, '%s.csv'%(cfg.subject))
# pd.DataFrame(sampled_ent).to_csv(save_path, index=False)
#%% 2. Sampling restricted by splited words
sampled_ent_ids = set(edss.sample_head_entity_for_edit(cfg.gen_n, rng, max_label_len = cfg.max_label_len))
sampled_ent = [ent for ent in edss.sub_ent if ent['id'] in sampled_ent_ids]
save_dir = os.path.join('data/wikidata/s9_edit_heads_sampling')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, '%s.csv'%(cfg.subject))
pd.DataFrame(sampled_ent).to_csv(save_path, index=False)
 