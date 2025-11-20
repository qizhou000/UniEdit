#%%
from uniedit_construction.tools.es_find_entity_tools import ESFindWikidataEntity
from elasticsearch_dsl import Search, Q
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
import json, os, re
import pandas as pd

esfwe = ESFindWikidataEntity()
csv_dir = 'data/wikidata/distribute_files_whole_claim'
esfwe.es_load_entity_from_csv_files('all_entity', True, csv_dir)
csv_dir = 'data/wikidata/filtered_entity/entity'
esfwe.es_load_entity_from_csv_files('filtered_entity', True, csv_dir)
