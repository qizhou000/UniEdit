#%%
from elasticsearch_dsl import Search, Q
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import List, Dict
from _io import TextIOWrapper
from tqdm import tqdm
import json, os, re
import pandas as pd

class ESFindWikidataEntity():
    def __init__(self, es_address = 'http://localhost:9200', print_stat = True):
        self.es = Elasticsearch([es_address], timeout = 300)
        print('ES Ping:', self.es.ping())
        if print_stat:
            self.es_stat()
    
    def es_stat(self):
        print('ES Stats:', self.es.cat.indices(format='json'))

    def es_load_entity_from_csv_files(self, index_name:str = 'test', 
            delete_old_index = True, csv_dir = 'data/wikidata/filtered_entity'): # complete
        def generate_actions(n):
            actions = []
            for d in data:
                document = dict(zip(columns, d))  
                actions.append({
                    "_op_type": "index",  
                    "_index": index_name,   
                    "_id": n,         #  ID
                    "_source": document   # 
                })
                n += 1
            return actions, n
        if delete_old_index and self.es.indices.exists(index = index_name):
            self.es.indices.delete(index = index_name)
        n = 1
        pbar = tqdm(position = 0, total = 113721)
        for root, dirs, files in os.walk(csv_dir):
            for f_name in files:
                data_path = os.path.join(root, f_name)
                data = pd.read_csv(data_path)
                data = data.where(pd.notnull(data), None) 
                columns = data.columns.tolist()
                data = data.values.tolist()
                actions, n = generate_actions(n)
                success, failed = bulk(self.es, actions)
                # print(success, failed)
                pbar.update(1)

    def es_large_amount_search(self, sc:Search, max_n:int, add_tqdm = True):
        res = sc.params(scroll = "5m", size = min(10000, max_n)).execute().to_dict()
        total_count = res['hits']['total']['value']
        max_score = res['hits']['max_score'] if total_count != 0 else -1
        results = [h['_source'] for h in res['hits']['hits']] if total_count != 0 else []
        scores = [h['_score'] for h in res['hits']['hits']] if total_count != 0 else []
        if len(results) == max_n or len(results) == total_count: 
            self.es.clear_scroll(scroll_id = res['_scroll_id'])
            return total_count, max_score, scores, results
        if add_tqdm:
            pbar = tqdm(position = 0, total = min(max_n, total_count), desc = 'ES large search')
            pbar.update(len(results))
        scroll_ids = [res['_scroll_id']]
        while len(results) < min(total_count, max_n):
            res = self.es.scroll(scroll_id = res['_scroll_id'], scroll="5m")
            scroll_ids.append(res['_scroll_id'])
            hits = res['hits']['hits']
            if len(hits) == 0:
                break
            if add_tqdm:
                pbar.update(len(hits))
            results.extend([h['_source'] for h in hits])
            scores.extend([h['_score'] for h in hits])
        self.es.clear_scroll(scroll_id = scroll_ids)
        return total_count, max_score, scores, results

    def es_multi_fields_match(self, doc:str, max_n = 100, index:str = 'all_entity', 
            fields:List[str] = ["labels", "descriptions", 'aliases'], add_tqdm = True):
        q = Q("multi_match", query = doc, fields=fields)
        sc = Search(using = self.es, index = index).query(q)
        total_count, max_score, scores, results = self.es_large_amount_search(sc, max_n, add_tqdm)
        return total_count, max_score, scores, results
    
    def es_multi_fields_docs_match(self, query_dict: Dict[str, List[str]], max_n=100, 
                            index='all_entity', add_tqdm=True):
        """query_dict: {
            "field1": [doc1, doc2, ...],
            "field2": [doc1, doc2, ...], ...
        }"""
        q = Q("bool", must=[Q("match", **{field: doc}) 
                        for field, docs in query_dict.items() for doc in docs])
        sc = Search(using = self.es, index = index).query(q)
        total_count, max_score, scores, results = self.es_large_amount_search(
            sc, max_n, add_tqdm)
        return total_count, max_score, scores, results

    def es_find_entity_by_ids(self, eids:List[str], index:str = 'all_entity', 
                              batch_size = 1000, add_tqdm = False):
        eids = [d.lower() for d in eids]
        results = []
        if add_tqdm:
            iters = tqdm(range(0, len(eids), batch_size), 'Finding entities')
        else:
            iters = range(0, len(eids), batch_size)
        for i in iters:
            q = Q("terms", id = eids[i:i+batch_size])
            size = min(batch_size, len(eids) - i)
            hits = Search(using = self.es, index = index).query(q).extra(size = size).execute()
            if len(hits) > 0: 
                results.extend([h.to_dict() for h in hits])
        return results

    def get_entity_with_distribute_position(self, p1:int, p2:int, p3:int, loaded_distribute_files_max_n = 4000): # half complete
        distribute_files_whole_claim = 'data/wikidata/distribute_files_whole_claim'
        if not hasattr(self, 'loaded_distribute_files'):
            self.loaded_distribute_files = {}
        if len(self.loaded_distribute_files) >= loaded_distribute_files_max_n:
            self.loaded_distribute_files = dict(sorted(self.loaded_distribute_files.items(), 
                key=lambda item: item[1][0])[loaded_distribute_files_max_n//2:])
        if (p1,p2) not in self.loaded_distribute_files:
            data_path = os.path.join(distribute_files_whole_claim, '%s000000'%p1, 
                        '%s.csv'%((p1 - 1)*1000000 + p2 * 1000))
            self.loaded_distribute_files[(p1,p2)] = [0, pd.read_csv(data_path)]
        self.loaded_distribute_files[(p1,p2)][0] += 1
        return self.loaded_distribute_files[(p1,p2)][1].iloc[p3].to_dict()
        # if f == 'json:
        #     if (p1,p2) not in self.loaded_distribute_files:
        #         distribute_files = 'data/wikidata/distribute_files'
        #         data_path = os.path.join(distribute_files, '%s000000'%p1, 
        #                     '%s.json'%((p1 - 1)*1000000 + p2 * 1000))
        #         with open(data_path, 'r') as f:
        #             self.loaded_distribute_files[(p1,p2)] = json.load(f)
        #     return self.loaded_distribute_files[(p1,p2)][p3]
    
    def get_dicts_from_latest_all_file(self, file:TextIOWrapper, start_p:int, 
                                min_size:int, search_chunk_size = 100000): # complete
        ''' Get entity dicts after the pointer `start_p` with min size `min_size` 
            from `file` latest-all.json file.'''
        def find_dict_boundary(sp, dict_start = True, min_size = 0):
            file.seek(sp)
            read_chunks = file.read(min_size)
            while True:
                new_chunk = file.read(search_chunk_size)
                if new_chunk == '': 
                    return -1, read_chunks
                read_chunks += new_chunk
                if dict_start: 
                    found_pos = read_chunks.find('\n{"type":"')
                else:
                    found_pos = read_chunks.find('},\n')
                    if found_pos == -1:
                        found_pos = read_chunks.find('}\n]')
                if found_pos != -1:
                    return found_pos + 1, read_chunks
        start_offset, _ = find_dict_boundary(start_p, True, 0)
        if start_offset == -1: return None
        end_offset, read_chunks = find_dict_boundary(start_p + start_offset, False, min_size)
        dicts_list = json.loads('[' + read_chunks[:end_offset] + ']')
        end_p = start_p + start_offset + end_offset
        return dicts_list, end_p

    def get_entity_from_latest_all_with_distribute_p(self, p1:int, p2:int, p3:int, 
                                        es_index = 'str_label_entity'): # complete
        latest_all_file_path = 'data/wikidata/latest-all.json'
        pos1 = (p1 - 1) * 1000000 + p2 * 1000 + p3
        left_pos, right_pos = 0, os.path.getsize(latest_all_file_path)
        with open(latest_all_file_path, 'r') as file:
            for i in range(42):
                mid_pos = int((left_pos + right_pos)//2)
                while True:
                    if mid_pos >= right_pos:
                        right_pos = int((left_pos + right_pos)//2)
                        mid_pos = int((left_pos + right_pos)//2)
                    dicts_list, end_p = self.get_dicts_from_latest_all_file(file, mid_pos, 0)
                    match_ent = dicts_list[0]
                    rrs = self.es_find_entity_by_ids([match_ent['id']], es_index)
                    if len(rrs) == 0:
                        mid_pos = end_p
                    else:
                        rrs = rrs[0]
                        break
                pos2 = (rrs['p1'] - 1) * 1000000 + rrs['p2'] * 1000 + rrs['p3']
                if pos1 > pos2:
                    left_pos = mid_pos
                elif pos1 < pos2:
                    right_pos = mid_pos
                else:
                    return match_ent 
        return None
