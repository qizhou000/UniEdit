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

fedg = FinalEditDataGenerator()
def judge_and_clean_final_data(d):
    if not fedg.has_valid_pt(d, ['edit']):
        return False
    for gl in ['generality', 'locality']:
        if len(d[gl]) > 1: raise # Only simply judge one
        if len(d[gl]) == 0: return False
        if '0' not in d[gl]: raise # Only simply judge one, and must '0'
        if not fedg.has_valid_pt(d, [gl, '0']):
            return False
    return True
arranged_data_dir = 'data/wikidata/s14_arranged_final_data'
save_data_dir = 'data/wikidata/s15_final_cleaned_data'
clean_data_dir = os.path.join(save_data_dir, 'cleaned')
os.makedirs(clean_data_dir, exist_ok = True)
unclean_data_dir = os.path.join(save_data_dir, 'other_unclean')
os.makedirs(unclean_data_dir, exist_ok = True)
all_subjects = ['biology', 'mathematics', 'chemistry', 'physics', 
    'geoscience', 'astronomy', 'sociology', 'jurisprudence', 
    'political science', 'economics', 'psychology', 'pedagogy', 
    'civil engineering', 'mechanical engineering', 'medicine', 
    'computer science', 'agronomy', 'literature', 'history', 
    'philosophy', 'art', 'material science', 'environmental science', 
    'sports science', 'data science']
for sub in tqdm(all_subjects, 'Cleaning final data'):
    with open(os.path.join(arranged_data_dir, sub, '0.json'), 'r') as f:
        data = json.load(f)
    clean_data = {}
    unclean_data = {}
    for k in data.keys():
        if judge_and_clean_final_data(data[k]):
            clean_data[k] = data[k]
        else:
            unclean_data[k] = data[k]
    with open(os.path.join(clean_data_dir, f'{sub}.json'), 'w') as f:
        json.dump(clean_data, f, indent = 4)
    with open(os.path.join(unclean_data_dir, f'{sub}.json'), 'w') as f:
        json.dump(unclean_data, f, indent = 4)
    print(f'Finished cleaning {sub} data')

#%% Stat data number
count = 0
for sub in tqdm(all_subjects):
    with open(os.path.join(clean_data_dir, f'{sub}.json'), 'r') as f:
        data = json.load(f)
    count += len(data)
print(count) 
