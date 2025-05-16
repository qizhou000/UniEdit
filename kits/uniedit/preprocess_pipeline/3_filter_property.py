#%%
import ijson, json, os
import pandas as pd
from tqdm import tqdm

dir_path = 'data/wikidata/property/all_property.json'
with open(dir_path, 'r') as f:
    data = json.load(f)
with open('data/wikidata/property/property_pages.json', 'r') as f:
    data_pages = json.load(f)
#%% filter by data type 
new_data = []
for d in data:
    if d['datatype'] in ['globe-coordinate', 'math', 'monolingualtext', 
        'quantity', 'string', 'time', 'wikibase-item']:  # 'musical-notation', 'wikibase-sense'
        new_data.append(d)
for d in new_data:
    d['# item pages'] = data_pages[d['id']]['# item pages']
new_data = sorted(new_data, key = lambda d:d['# item pages'], reverse = True)
len(new_data)
#%% directly filter
filter_props = set([
    # Artificial filter
    "P2860", "P1545", "P478",
    # index 
    "P4895", "P8875", "P1685", 
    # no english description
    "P3592","P5804","P2201","P6095","P2490","P2231","P6069","P5244","P6793","P2329","P10636",
    # wikimedia 
    "P143","P373","P5008","P301","P910","P935","P1472","P1753","P1754","P7867","P2354","P1792","P6112","P7084","P2959","P3096","P2517","P1612","P424","P1791","P6568","P4195","P3858","P1200","P9926","P1204","P1151","P1881","P5125","P8464","P6186","P2307","P8625","P10408","P12692","P6731","P12503","P10358","P12505",
    # wiki 
    "P11797","P1813","P6104","P2096","P1114","P9675","P2302","P4241","P1855","P1793","P4070","P2875","P4329","P2377","P5869","P4775","P7383","P3734","P3713","P3709","P4680","P2719","P5977","P8274","P2803","P6695","P10714",
    # id 
    "P9675","P593","P2322","P2676","P1748","P554","P4776","P711","P12883","P667","P1931","P799","P12889","P9410","P6954","P8687","P9660","P8966","P11136","P12463",
    # page
    "P304","P1104","P7668","P4714","P10999","P8330","P11627",
    # time
    "P813","P5017","P2960","P729","P1191",
    # monolingualtext
    "P1476","P1448","P1705","P1843","P1559","P2561","P1683","P1477","P1680","P2795","P1922","P2521","P5187","P1684","P1449","P3321","P1549","P6208","P5831","P5168","P3132","P1451","P2441","P8394","P9376","P8338","P2562","P2559","P7535","P9570","P6833","P6333","P7081","P2275","P2916","P1635","P1638","P6251","P6607","P4239","P9767","P6427","P7008","P9533","P6346","P8898","P3909","P7150","P2315","P8770","P10655","P1450","P12089","P11265","P12090",
    # table
    "P8558",
    # code
    "P4575","P1257","P487","P4213","P5522","P9382","P223","P5519","P281","P6375","P3295","P240", "P239", "P10627", "P1692", "P3921", "P11957", "P3441", "P10564", "P8283", "P10604", "P1987", "P5949", "P1068", "P2859", "P2179", "P1611", "P1796", "P952", "P3994", "P2357", "P877", "P7290", "P10643", "P5591", "P3625", "P3067", "P6733", "P5518", "P7315", "P874", "P876", "P3969", "P11323", "P12434",
    # street
    "P5423","P1282","P5423","P6592",
    # catalog
    "P7328","P4876","P9969","P8615",
    # # item pages <= 3
    "P1917","P6604","P1913","P1915","P4850","P1914","P10637","P9052","P9275","P13150","P10734","P4078","P8789","P10733","P9310","P9126","P10746","P5439","P5802","P10737","P3775","P3778","P3779","P4325","P8276","P6212",
    # filter by LLM
    "P528","P1810","P7793","P217","P1351","P348","P1618","P1326","P1319","P3879","P3878","P1329","P958","P393","P473","P490","P670","P1588","P4970","P4856","P395","P3903","P2900","P3294","P1824","P4243","P296","P617","P1945","P2364","P792","P7421","P2598","P1671","P465","P249","P529","P598","P1998","P2802","P238","P9759","P10135","P7125","P8746","P8745","P11327","P230","P11916","P426","P6524","P3113","P4649","P8477","P1471","P6875","P7338","P5778","P9994","P6883","P8470","P2258","P3970","P11106","P944","P5703","P8498","P2259","P12275","P1641","P913","P546","P2560","P4573","P10703","P5810","P1360","P5461","P2285","P5471","P847","P5994","P7126","P875","P5575","P5608","P2223","P10681","P1295","P2717","P4269","P6073","P3486","P5668","P5481","P5706","P1689","P3251","P10623","P4851","P5992","P8465","P10273","P5672","P3252","P5947","P8466","P10316","P5709","P4242","P4105","P2222","P2662","P8461","P2294","P2296","P5682","P8275","P4445","P4268","P5893","P2718","P5685","P7083","P5678","P2220","P433","P972","P1932","P1343","P10649",
    # Sandbox
    "P1106", "P370", "P369", "P5979", "P578", "P2535",
    # coordinate
    "P1240",
    # sentence meanless
    "P225", "P31", "P1889"
])
new_data = [d for d in new_data if d['id'] not in filter_props]
len(new_data)
#%% Artificial pattern filter
import re
dtps = set()
dd = []
for d in new_data:
    if type(d['descriptions']) == dict:
        pass
    elif re.search(r'coordi', d['labels'], re.IGNORECASE):
        pass
        print('"%s"'%d['id'], end=', ')
        print(d['labels'], '\n', d['descriptions'], end='\n')
        print(d['# item pages'], end='\n\n')
        dd.append(d)
    elif re.search(r'coordi', d['descriptions'], re.IGNORECASE):
        pass
        print('"%s"'%d['id'], end=',')
        print(d['labels'], '\n', d['descriptions'], end='\n')
        print(d['# item pages'], end='\n\n')
        dd.append(d)
    elif d['datatype'] == 'quantity':
        pass
        # print('"%s"'%d['id'], end=',')
        # print(d['id'], d['labels'], d['# item pages'], '\n', d['descriptions'], end='\n\n')
    elif d['# item pages'] <= 3:
        pass
        # print('"%s"'%d['id'], end=',')
        # print(d['id'], d['labels'], d['# item pages'], '\n', d['descriptions'], end='\n\n')
    dtps.add(d['datatype'])
    # print(d['id'], d['labels'], d['# item pages'], '\n', d['descriptions'], end='\n\n')
    # print(d['id'], d['labels'], d['# item pages'], end='\n\n')
len(dd)
#%% filter by LLM
i = 82
batch_size = 30
batch_d = new_data[i*batch_size:(i+1) * batch_size]
for d in batch_d:
    print('%s: %s\n%s'%(d['id'], d['labels'], d['descriptions']))
    print()
new_data1 = [d for d in new_data if d['id'] not in filter_props]
#%% save
with open('data/wikidata/property/filtered_property.json', 'w') as f:
    json.dump(new_data, f, indent=4)
new_data_table = [[d['id'], d['labels'], d['# item pages']] for d in new_data]
pd.DataFrame(new_data_table, columns = [['id', 'label', '# item pages']]
 ).to_csv('data/wikidata/property/filtered_property.csv', index=False)
#%% plot bar
import matplotlib.pyplot as plt

item_pages = [d[2] for d in new_data_table]
plt.bar(range(len(item_pages)), item_pages)
plt.show()

all_pages = sum(item_pages)
accum_item_pages = [new_data_table[0][2]/all_pages]
for d in new_data_table[1:]:
    accum_item_pages.append(accum_item_pages[-1] + d[2]/all_pages)
plt.bar(range(len(item_pages)), accum_item_pages)
plt.show()
