#%%
import ijson, json, os
from tqdm import tqdm

filename = 'data/wikidata/latest-all.json'
#%%
from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)  
        return super().default(obj)

def modidy_item(item:dict):
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

json_n = 0
item_list = []
batch_size = 1000
save_dir = 'data/wikidata/all_property'
os.makedirs(save_dir, exist_ok = True)
with open(filename, 'rb') as f:
    objects = ijson.items(f, 'item')  
    for obj in tqdm(objects, 'Search all properties', total = 1143721000):
        if 'Q' in obj['id']:
            continue
        item_list.append(modidy_item(obj))
        print('Now batch-%s-%s'%(json_n, len(item_list)))
        if len(item_list) == batch_size: 
            json_n += len(item_list)
            with open(os.path.join(save_dir, '%s.json'%(json_n)), 'w') as f:
                json.dump(item_list, f, cls = DecimalEncoder, indent = 4)
            item_list = []
if len(item_list) != 0:
    json_n += len(item_list)
    with open(os.path.join(save_dir, '%s.json'%(json_n)), 'w') as f:
        json.dump(item_list, f, cls = DecimalEncoder, indent = 4)

####################################################################################
####################################################################################
# After the above, integrate the scattered property files together and save them in 
# `data/wikidata/property/all property/all_propellant.json`. 
# Then, copy <quantity of item pages> for all properties from the website 
# https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/1-1000
# and save them to `data/wikidata/property/property_pages.csv` and 
# `data/wikidata/property/property_pages.json`, used for subsequent property filtering.
####################################################################################
####################################################################################

