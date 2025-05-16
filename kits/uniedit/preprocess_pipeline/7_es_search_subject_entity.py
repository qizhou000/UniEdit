#%%
from kits.uniedit.tools.es_find_entity_tools import ESFindWikidataEntity
from collections import Counter
import json, os, re, argparse
from tqdm import tqdm
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--subject', type=str, required=True)
    return parser.parse_args()
class cfg:
    subject = 'biology'
cfg = get_args()
subject = cfg.subject
esfwe = ESFindWikidataEntity()
#%% Search subject entity with ES and save
with open('data/wikidata/subject_words.json', 'r') as f:
    subject_words = list(set(json.load(f)[subject]))
save_root = 'data/wikidata/s7_subject_entity_es'
doc = ' '.join(subject_words)
total_count, max_score, scores, results = esfwe.es_multi_fields_match(
    doc, max_n = 9999999999, index = 'filtered_entity', 
    fields = ["labels", "descriptions", 'aliases'])
save_res = [[r['id'], r['labels'], r['descriptions'], s, r['claim_n'], r['claims']] 
            for r, s in zip(results, scores)]
save_path = os.path.join(save_root, '%s.csv'%subject)
pd.DataFrame(save_res, columns=['id', 'labels', 'description', 'retr_score', 
                        'claim_n', 'claims']).to_csv(save_path, index=False)
