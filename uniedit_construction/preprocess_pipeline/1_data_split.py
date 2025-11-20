#%%
import ijson, json, os
from tqdm import tqdm
import pandas as pd
from decimal import Decimal

filename = 'data/wikidata/latest-all.json'

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)  
        return super().default(obj)

def modidy_item(item:dict, item_keys:list):
    for k in list(item.keys()):
        if k not in item_keys:
            item.pop(k)
    try: item['labels'] = item['labels']['en']['value']
    except: pass
    try: item['descriptions'] = item['descriptions']['en']['value']
    except: pass
    try: item['aliases'] = [i['value'] for i in item['aliases']['en']]
    except: pass
    # try:
    #     claims_keys = ['mainsnak', 'type']
    #     for prop, prop_vs in item['claims'].items():
    #         item['claims'][prop] = [{k: v[k] for k in claims_keys} for v in prop_vs]
    # except: pass
    return item
def split_data(save_root, batch_size = 1000, dir_size = 1000000, save_format = 'csv'):
    item_keys = ['type', 'id', 'labels', 'descriptions', 'aliases', 
                  'claims', 'ns', 'modified']
    item_list = []
    pbar = tqdm(total=113721000, dynamic_ncols = True)
    with open(filename, 'rb') as f:
        objects = ijson.items(f, 'item')  
        for i, obj in enumerate(objects):
            item  = modidy_item(obj, item_keys)
            item_list.append(item)
            if (i + 1) % batch_size == 0: 
                save_dir = os.path.join(save_root, str((i//dir_size + 1) * dir_size))
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                if save_format == 'csv':
                    df = pd.DataFrame(item_list, columns = item_keys)
                    df.to_csv(os.path.join(save_dir, '%s.csv'%(i+1)), index=False)
                else:
                    with open(os.path.join(save_dir, '%s.json'%(i+1)), 'w') as f:
                        json.dump(item_list, f, cls = DecimalEncoder, indent = 4)
                item_list = []
            pbar.update()

save_root = 'data/wikidata/distribute_files_whole_claim'
split_data(save_root, batch_size = 1000, dir_size = 1000000, save_format = 'csv')
