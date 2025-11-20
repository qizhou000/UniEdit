#%%
from collections import Counter
import json, os, re
import pandas as pd
from tqdm import tqdm

descriptions = []
data_root = 'data/wikidata/label_filtered_entity'
for file_name in tqdm(os.listdir(data_root)):
    data_path = os.path.join(data_root, file_name)
    if not os.path.isfile(data_path): continue
    data = pd.read_csv(data_path)
    descriptions.extend(data['descriptions'].tolist())
#%% stat
frequency = Counter(descriptions)
most_common = frequency.most_common()
most_common = [d for d in most_common if not pd.isna(d[0])]
frequency_dict = dict(frequency)
df = pd.DataFrame(most_common, columns=['Description', 'frequency'])
df.to_csv('data/wikidata/description/all_descriptions.csv', index=False)
#%% read
import json, os, re
import pandas as pd
from tqdm import tqdm

description_path = 'data/wikidata/description/all_descriptions.csv'
most_common = pd.read_csv(description_path).values.tolist()
#%% filter
whole_match = set([
    "series in the National Archives and Records Administration's holdings",
    "item in the National Archives and Records Administration's holdings",
    "point of time",
    "Unicode character",
    "CJK (hanzi/kanji/hanja) character",
    "prime number",
    "file unit in the National Archives and Records Administration's holdings",
    "entry in Alumni Oxonienses: the Members of the University of Oxford, 1715-1886",
    "Christian hymn text",
    "entry in Dictionary of National Biography",
    "hymn tune",
    "shipwreck off the Scottish coast",
    "United States Supreme Court case", 
    "doctoral thesis",
    "Hangul syllable",
    "file format",
    "watercourse in Democratic Republic of the Congo",
    "cross-reference in Paulys Realencyclopädie der classischen Altertumswissenschaft (RE)",
    "person identified in the Museu da Pessoa collection",
    "Wikimédia category",
    "request for comments publication",
    "year",
])
regular_match = [
    r'([^p]|^)article',
    r'(primary|junior high|high|secondary|middle|vocational) school', 
    r'^school in',
    r'\breport(s|age)?\b',
    r'branch[a-z ]+bank',
    r'street(s|light)? (in|of)',
    r'\bwikimedia\b',
    r'\bdate\b(:| in)',
    r"(given|family) name",
    r'(season (of|[0-9]+)|(team|ball|hockey|soccer|sports|club|cricket|tournament|league|competition|tour) season)'
]
str_match = [
    r"watercourse",
    r"'language':",
    r"{}",
    r'bank in',
    r'bank branch',
    r'championship', 
    r'wikimédia',
    r'template'
]
# filter
now_common = []
for desc, ct in tqdm(most_common):
    if desc in whole_match: continue
    flg = False
    for m in regular_match:
        if re.search(m, desc, re.IGNORECASE):
            flg = True
            break
    if flg: continue
    for m in str_match:
        if m in desc.lower():
            flg = True
            break
    if flg: continue
    now_common.append((desc, ct))
#%% check
i, n = 0, 30
now_common[i*30:(i+1)*30], len(now_common), sum([c[1] for c in now_common])

#%% save
df = pd.DataFrame(now_common, columns=['Description', 'frequency'])
df.to_csv('data/wikidata/description/filtered_descriptions.csv', index=False)
#%% 
# "(':', ""'наукова"")",2148760
# "(""'наукова"", 'стаття')",2148743
# "(',', ""'de"")",2036062
# "('стаття', ',')",1852868
# "(',', 'опублікована')",1852868
# "('in', 'the')",1829450
# "(':', ""'wetenschappelijk"")",1795631
# "('1', ',')",1783898
# "(""'wetenschappelijk"", 'artikel')",1765401
# "(""'ar"", ""'"")",1639936
# "(':', ""'fr"")",1565193
# "('found', 'in')",1484135
# "(':', ""'article"")",1281476
# "('chemical', 'compound')",1269284
# "('опублікована', 'в')",1215030
# "(':', ""'im"")",1200873
# "('wissenschaftlicher', 'artikel')",1200615
# "('veröffentlichter', 'wissenschaftlicher')",1195444
# "(""'article"", 'scientifique')",1155909
str_match = [
    ": {'language':",
]
regular_match = [
    
]
exact_match = set([
]) # 'season'

descs = []
for k in tqdm(most_common):
    try:
        if 'wetenschappelijk' in k[0].lower():
            if k[0] in exact_match:
                continue
            flg = True
            for m in str_match:
                if m in k[0].lower(): 
                    flg = False
                    break
            if not flg: continue
            for m in regular_match:
                if re.search(m, k[0], re.IGNORECASE):
                    flg = False
                    break
            if flg: descs.append(k)
    except: pass
#%% 
i, n = 0, 30
print(descs[i*n:(i+1)*n])
print(len(descs), sum([d[1] for d in descs]))
