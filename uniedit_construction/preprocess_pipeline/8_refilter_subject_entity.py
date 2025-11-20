#%% Refiter subject entity with subject words. 
# The entity who has words or phrase in label or description shared with the set of subject words will be saved.
from nltk.tokenize import word_tokenize
from collections import Counter
import json, os, re, argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import string
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--subject', type=str, required=True)
    return parser.parse_args()
args = get_args()
subject = args.subject # args.subject # sports science/philosophy/literature/physics/civil engineering/economics/agronomy/astronomy/jurisprudence/pedagogy
with open('data/wikidata/subject_words.json', 'r') as f:
    subject_words = set(json.load(f)[subject])
sub_ent_path = os.path.join('data/wikidata/s7_subject_entity_es', '%s.csv'%subject)
subject_entity = pd.read_csv(sub_ent_path).to_dict('records')
#%%
def include_subject_words(ent):
    sentence = ent['labels'].lower() + ' ' + ent['description'].lower()
    for sws in subject_words:
        if re.search(r'\b'+ sws.lower() + r'\b', sentence):
            return True
    return False
subject_words_split = set([w.lower() for sw in subject_words for w in sw.split(' ')]) 
if '' in subject_words_split: 
    subject_words_split.remove('')
subject_match_ents = []
for ent in tqdm(subject_entity):
    if not isinstance(ent['labels'], str) or not isinstance(ent['description'], str):
        continue
    if not include_subject_words(ent): 
        continue
    label_tokens = word_tokenize(ent['labels'].lower())
    des_tokens = word_tokenize(ent['description'].lower())
    ent_words = set([w for w in [*label_tokens, *des_tokens] if w not in string.punctuation])
    sub_words_intersect = subject_words_split & ent_words
    if len(sub_words_intersect) > 0:
        ent['share_sub_words_n'] = len(sub_words_intersect)
        ent['share_sub_words'] = list(sub_words_intersect)
        non_sub_words = ent_words - sub_words_intersect
        ent['non_sub_words_n'] = len(non_sub_words)
        ent['non_sub_words'] = list(non_sub_words)
        subject_match_ents.append(ent)
#%%
save_dir = 'data/wikidata/s8_subject_entity'
if not os.path.exists(save_dir): 
    os.makedirs(save_dir) 
save_path = os.path.join(save_dir, '%s.csv'%subject)
d = pd.DataFrame(subject_match_ents).to_csv(save_path, index=False)
