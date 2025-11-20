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

cleaned_data_dir = 'data/wikidata/s15_final_cleaned_data/cleaned'
save_root = 'data/UniEdit'
os.makedirs(os.path.join(save_root, 'train'), exist_ok=True)
os.makedirs(os.path.join(save_root, 'test'), exist_ok=True)
rng = np.random.default_rng(1234)
train_data_propo = 2/3
for data_name in tqdm(os.listdir(cleaned_data_dir)):
    data_path = os.path.join(cleaned_data_dir, data_name)
    with open(data_path, 'r') as f:
        data = json.load(f)
    data_keys = list(data.keys())
    train_data = {k: data[k] for k in data_keys[:int(len(data_keys) * train_data_propo)]}
    test_data = {k: data[k] for k in data_keys[int(len(data_keys) * train_data_propo):]}
    with open(os.path.join(save_root, 'train', data_name), 'w') as f:
        json.dump(train_data, f, indent = 4) 
    with open(os.path.join(save_root, 'test', data_name), 'w') as f:
        json.dump(test_data, f, indent = 4) 
