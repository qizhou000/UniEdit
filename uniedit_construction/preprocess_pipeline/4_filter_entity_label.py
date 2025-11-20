#%%
import json, os
import pandas as pd
from tqdm import tqdm

def save_csv(ents, ents_n):
    w_path = 'data/wikidata/label_filtered_entity/%s.csv'%ents_n
    pd.DataFrame(ents, columns=['id', 'label_lan', 'labels', 'descriptions', 
                                'p1', 'p2', 'p3']).to_csv(w_path, index=False)

data_root = 'data/wikidata/distribute_files'
max_n = 113721670
pbar = tqdm(total = max_n, dynamic_ncols = True)
ents = []
ents_n = 0
ents_max_n = 1000000
for i in range(1, 115):
    data_dir = os.path.join(data_root, '%s000000'%i)
    for j in range(1, 1001):
        m = (i - 1) * 1000000 + j * 1000
        data_path = os.path.join(data_dir, '%s.json'%m)
        if not os.path.exists(data_path): break
        with open(data_path, 'r') as f: 
            data = json.load(f)
        for k, d in enumerate(data):
            label_lan = 'en'
            if type(d['labels']) == dict:
                if 'en' in d['labels']: 
                    d['labels'] = d['labels']['en']['value']
                elif 'mul' in d['labels']: 
                    label_lan = 'mul'
                    d['labels'] = d['labels']['mul']['value']
            if type(d['labels']) == str:
                apdd = [d['id'], label_lan, d['labels'], d['descriptions'], 
                        str(i), str(j), str(k)]
                ents.append(apdd)
                if len(ents) % ents_max_n == 0:
                    ents_n += ents_max_n
                    save_csv(ents, ents_n)
                    ents = []
            pbar.update(1)
if len(ents) != 0:
    ents_n += len(ents)
    save_csv(ents, ents_n)

