#%% 
import matplotlib.pyplot as plt
from typing import List, Dict
import os, pandas, json
import pandas as pd
import numpy as np
import warnings
# warnings.simplefilter("error", category=RuntimeWarning)
warnings.simplefilter("default", category=RuntimeWarning)
#%%
read_eval_results = {}
#%%
def get_values(patterns:Dict, # True/False/-1
            editor_name:str, model_name:str, metric:str, topk = 5, seed = None):
    eval_res_name = '%s-%s-seed-%s'%(editor_name, model_name, seed) # read editor-model json file
    if eval_res_name not in read_eval_results:
        if seed == None:
            res_path = f'eval_results/llms/{editor_name}/{model_name}/UNIEDIT/sequential_edit_1/results.json'
        else:
            res_path = f'eval_results/llms/{editor_name}/{model_name}/UNIEDIT/sequential_edit_1/seed_{seed}_results.json'
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                read_res = json.load(f)
                read_eval_results[eval_res_name] = [d[0] for d in read_res]
    if eval_res_name not in read_eval_results:
        return []
    def if_valid_pattern_rd(rd):
        if metric == 'gen':
            for p in rd['generality']['uni_gen'][0]['patterns'].keys():
                if patterns[p] == -1: continue
                assert isinstance(patterns[p], bool)
                if patterns[p] != rd['generality']['uni_gen'][0]['patterns'][p]:
                    return False
        elif metric == 'loc':
            if len(rd['locality']['uni_loc']) == 0:
                return False
            for p in rd['locality']['uni_loc'][0]['patterns'].keys():
                if patterns[p] == -1: continue
                assert isinstance(patterns[p], bool)
                if patterns[p] != rd['locality']['uni_loc'][0]['patterns'][p]:
                    return False
        else: raise
        return True
    res_data = [rd for rd in read_eval_results[eval_res_name] if if_valid_pattern_rd(rd)]
    if metric == 'gen':
        all_acc = []
        for d in res_data:
            gd = d['generality']['uni_gen'][0]
            tok_in = [tt in pt[:topk] for pt, tt in zip(
                gd['post_edit_predict_tokens'], gd['target_tokens'])]
            all_acc.append(sum(tok_in) / len(tok_in))
    elif metric == 'loc':
        all_acc = []
        for d in res_data:
            ld = d['locality']['uni_loc'][0]
            tok_in = [tt in pt[:topk] for pt, tt in zip(
                ld['post_edit_predict_tokens'], ld['pre_edit_predict_tokens'])]
            all_acc.append(sum(tok_in) / len(tok_in))
    else: raise
    return all_acc

patterns = {
    'gen': {
        "Rep": {"rephrase": True, "multi-hop": False, "relation reverse": False,
            "same entity reason": False, "subject alias": False, "object alias": False},
        "RR": {"rephrase": False, "multi-hop": False, "relation reverse": True,
            "same entity reason": False, "subject alias": False, "object alias": False},
        "MH": {"rephrase": False, "multi-hop": True, "relation reverse": False,
            "same entity reason": False, "subject alias": False, "object alias": False},
        "SA": {"rephrase": True, "multi-hop": False, "relation reverse": False,
            "same entity reason": False, "subject alias": True, "object alias": False},
        "OA": {"rephrase": True, "multi-hop": False, "relation reverse": False,
            "same entity reason": False, "subject alias": False, "object alias": True},
        "SER": {"rephrase": False, "multi-hop": False, "relation reverse": False,
            "same entity reason": True, "subject alias": False, "object alias": False},

        "SER,RR": {"rephrase": False, "multi-hop": False, "relation reverse": True,
            "same entity reason": True, "subject alias": False, "object alias": False},
        "RR,MH,OA": {"rephrase": False, "multi-hop": True, "relation reverse": True,
            "same entity reason": False, "subject alias": False, "object alias": True},
        "RR,MH,OA,SA": {"rephrase": False, "multi-hop": True, "relation reverse": True,
            "same entity reason": False, "subject alias": True, "object alias": True},
        "MH,OA": {"rephrase": False, "multi-hop": True, "relation reverse": False,
            "same entity reason": False, "subject alias": False, "object alias": True},
        "RR,OA": {"rephrase": False, "multi-hop": False, "relation reverse": True,
            "same entity reason": False, "subject alias": False, "object alias": True},
        "Rep,OA,SA": {"rephrase": True, "multi-hop": False, "relation reverse": False,
            "same entity reason": False, "subject alias": True, "object alias": True},
    },
    'loc': {
        "W/O": {"multi-hop": False, "same entity reason": False, "subject specificity": False,
            "relation specificity": False, "object specificity": False, "1-N forgotten": False},
        "OS": {"multi-hop": False, "same entity reason": False, "subject specificity": False,
            "relation specificity": False, "object specificity": True, "1-N forgotten": False},
        "RS": {"multi-hop": False, "same entity reason": False, "subject specificity": False,
            "relation specificity": True, "object specificity": False, "1-N forgotten": False},
        "SS": {"multi-hop": False, "same entity reason": False, "subject specificity": True,
            "relation specificity": False, "object specificity": False, "1-N forgotten": False},
        "1-NF": {"multi-hop": False, "same entity reason": False, "subject specificity": -1,
            "relation specificity": -1, "object specificity": -1, "1-N forgotten": True},


        "1-NF,MH": {"multi-hop": False, "same entity reason": False, "subject specificity": -1,
            "relation specificity": -1, "object specificity": -1, "1-N forgotten": True},
        "SS,MH": {"multi-hop": True, "same entity reason": False, "subject specificity": True,
            "relation specificity": False, "object specificity": False, "1-N forgotten": False},
        "RS,MH": {"multi-hop": True, "same entity reason": False, "subject specificity": False,
            "relation specificity": True, "object specificity": False, "1-N forgotten": False},
        "OS,RS": {"multi-hop": False, "same entity reason": False, "subject specificity": False,
            "relation specificity": True, "object specificity": True, "1-N forgotten": False},
        "MH": {"multi-hop": True, "same entity reason": False, "subject specificity": False,
            "relation specificity": False, "object specificity": False, "1-N forgotten": False},
    } 
}
editor_names = ['woe', 'ft', 'ike', 'rome', 'serac', 'tp', 'grace', 'alphaedit'] 
metrics = ['gen', 'loc'] # , 'loc'
results = {}
for mtc in metrics:
    results[mtc] = {}
    for model_name in ['gpt2-xl', 'gpt-j-6b', 'llama-3-8b']:
        results[mtc][model_name] = {}
        for editor_name in editor_names:
            results[mtc][model_name][editor_name] = {}
            for p_name, p in patterns[mtc].items():
                vs0 = get_values(p, editor_name, model_name, mtc, 5, seed = None)
                vs1 = get_values(p, editor_name, model_name, mtc, 5, seed = 8)
                vs2 = get_values(p, editor_name, model_name, mtc, 5, seed = 16)
                vs3 = get_values(p, editor_name, model_name, mtc, 5, seed = 64)
                vs = [*vs0, *vs1, *vs2, *vs3]
                results[mtc][model_name][editor_name][p_name] = np.mean(vs)*100

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

plt.rcParams['font.family'] = 'Times New Roman'
def plot_radar_chart(data, labels, model_names=None, angle_offset=0, 
                     save_path=None, save_formats=('svg',), font_size=23):
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  

    angle_offset_rad = np.deg2rad(angle_offset)
    angles = [(a + angle_offset_rad) % (2 * np.pi) for a in angles]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, model_data in enumerate(data):
        values = model_data + model_data[:1]
        label = model_names[i] if model_names else f'Model {i + 1}'
        ax.plot(angles, values, label=label)
        ax.fill(angles, values, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=font_size)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=font_size)
    # ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=font_size)
    
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        base, _ = os.path.splitext(save_path)
        for fmt in save_formats:
            plt.savefig(f"{base}.{fmt}", format=fmt, transparent=True)
    plt.show()

editor_name_map = {'woe':'W/O', 'tp':'T-Patcher', 'alphaedit': 'AlphaEdit'}
for mtc in results.keys():
    for model_name in results[mtc]:
        editor_names = [editor_name for editor_name in results[mtc][model_name]]
        labels = [p for p in results[mtc][model_name][editor_names[0]]] 
        data = [[results[mtc][model_name][editor_name][l] for l in labels] 
                for editor_name in editor_names]
        editor_names = [editor_name_map[editor_name] if editor_name in 
            editor_name_map else editor_name.upper() for editor_name in editor_names]
        plot_radar_chart(data, labels, editor_names, 
            angle_offset=105 if mtc == 'gen' else 6*18, 
            save_path = f'figs/uniedit/performance_patterns/{mtc}-{model_name}.svg')

# path = 'figs/uniedit/rader_eval_pattern_results'
# plot_radar_chart(data, labels, model_names, angle_offset=0)
