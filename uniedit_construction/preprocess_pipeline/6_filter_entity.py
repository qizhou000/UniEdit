#%%
import json, os, re
import pandas as pd
from tqdm import tqdm
filtered_property_path = 'data/wikidata/property/filtered_property.csv'
filtered_property = set(pd.read_csv(filtered_property_path)['id'].tolist())
filtered_description_path = 'data/wikidata/description/filtered_descriptions.csv'
filtered_description = set(pd.read_csv(filtered_description_path)['Description'].tolist())
#%% filter entity
def save_csv(ents_n, ents):
    dir_path = 'data/wikidata/filtered_entity/entity'
    os.makedirs(dir_path, exist_ok = True)
    w_path = os.path.join(dir_path, '%s.csv'%ents_n)
    pd.DataFrame(ents, columns=['type', 'id', 'labels', 'descriptions', 
        'aliases', 'ns', 'modified', 'p1', 'p2', 'p3', 'claim_n', 'claims']).to_csv(w_path, index=False)

def valid_entity(ent):
    label, description = ent['labels'], ent['descriptions']
    # description
    if type(description) != str: # filter dict that no english description
        return False
    if description not in filtered_description:
        return False
    # label
    if type(label) != str: # filter dict
        return False
    if re.search(r'^[0-9]*$', label): # re.search(r'^[^a-z]*$', label):
        return False
    return True

data_root = 'data/wikidata/distribute_files'
ents = []
ents_n = 0
ents_max_n = 1000000
pbar = tqdm(total=113721670, initial=0, position=0)
for i in range(1, 115):
    data_dir = os.path.join(data_root, '%s000000'%i)
    for j in range(1, 1001):
        data_path = os.path.join(data_dir, '%s.json'%((i - 1) * 1000000 + j * 1000))
        if not os.path.exists(data_path): break
        with open(data_path, 'r') as f: data = json.load(f)
        for k, d in enumerate(data):
            if valid_entity(d):
                # filter property
                claims = [c for c in d['claims'].keys() if c in filtered_property]
                # add entity
                apdd = [d['type'], d['id'], d['labels'], d['descriptions'], 
                    d['aliases'], d['ns'], d['modified'], i, j, k,
                    len(claims), claims]
                ents.append(apdd)
                if len(ents) % ents_max_n == 0:
                    ents_n += ents_max_n
                    save_csv(ents_n, ents)
                    ents = []
            pbar.update(1)
if len(ents) != 0:
    ents_n += len(ents)
    save_csv(ents_n, ents)

#%% seperately store filtered entity id
import os
import pandas as pd
load_dir = 'data/wikidata/filtered_entity/entity'
ent_ids = []
for csv_name in tqdm(os.listdir(load_dir)):
    load_path = os.path.join(load_dir, csv_name)
    d = pd.read_csv(load_path)
    ent_ids.extend(d['id'])
    print(len(ent_ids))
save_path = os.path.join('data/wikidata/filtered_entity', 'only_id.json')
with open(save_path, 'w') as f:
    json.dump(ent_ids, f)

