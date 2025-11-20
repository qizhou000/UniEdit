#%% 
import matplotlib.pyplot as plt
import os, pandas, json
from typing import List
import pandas as pd
import numpy as np
 
read_eval_results = {}
#%%
def get_values(disciplines:List[str], editor_name:str, model_name:str, 
            metric, topk = 5, patterns_not = [], seed = None):
    eval_res_name = '%s-%s-seed-%s'%(editor_name, model_name, seed) # read editor-model json file
    if eval_res_name not in read_eval_results:
        if seed == None:
            res_path = f'eval_results/llms/{editor_name}/{model_name}/UNIEDIT/sequential_edit_1/results.json'
        else:
            res_path = f'eval_results/llms/{editor_name}/{model_name}/UNIEDIT/sequential_edit_1/seed_{seed}_results.json'
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                read_res = json.load(f)
                res_data = {}
                for d in read_res:
                    d = d[0]
                    dis = d['index'].split('-')[0]
                    if dis not in res_data: 
                        res_data[dis] = []
                    res_data[dis].append(d)
                read_eval_results[eval_res_name] = res_data
    if eval_res_name not in read_eval_results:
        return []
    res_data = []
    for dis in disciplines: # filter disciplines
        res_data.extend(read_eval_results[eval_res_name][dis])
    def if_valid_pattern_rd(rd):
        if metric == 'rel':
            return True
        if metric == 'gen':
            for pm in patterns_not:
                if rd['generality']['uni_gen'][0]['patterns'][pm]:
                    return False
        elif metric == 'loc':
            if len(rd['locality']['uni_loc']) == 0:
                return False
            for pm in patterns_not:
                if rd['locality']['uni_loc'][0]['patterns'][pm]:
                    return False
        return True
    res_data = [rd for rd in res_data if if_valid_pattern_rd(rd)]
    if metric == 'rel':
        all_acc = [d['reliability'][0]['acc'] for d in res_data]
    elif metric == 'gen':
        all_acc = []
        for d in res_data:
            gd = d['generality']['uni_gen'][0]
            tok_in = [tt in pt[:topk] for pt, tt in zip(
                gd['post_edit_predict_tokens'], gd['target_tokens'])]
            all_acc.append(sum(tok_in) / len(tok_in))
    elif metric == 'loc':
        all_acc = []
        for d in res_data:
            if len(d['locality']['uni_loc']) == 0:
                continue
            ld = d['locality']['uni_loc'][0]
            tok_in = [tt in pt[:topk] for pt, tt in zip(
                ld['post_edit_predict_tokens'], ld['pre_edit_predict_tokens'])]
            all_acc.append(sum(tok_in) / len(tok_in))
    else:
        raise
    return all_acc

editor_names = ['woe', 'ft', 'ike', 'rome', 'serac', 'tp', 'grace', 'alphaedit'] 
all_subjects = {
    'Nat. Sci.': ['astronomy', 'biology', 'chemistry', 'geoscience', 
                        'mathematics', 'physics'],
    'Human.': ['art', 'history', 'literature', 'philosophy'],
    'Soc. Sci.': ['economics', 'jurisprudence',
        'pedagogy', 'political science', 'psychology', 'sociology'],
    'App. Sci.': ['agronomy', 'civil engineering',
        'computer science', 'mechanical engineering', 'medicine'],
    'Inter. Stu.': ['data science', 'environmental science',
        'material science', 'sports science']
}
metric = 'gen' # rel/gen/loc
patterns_not = {
    'rel': [],
    'gen': ['multi-hop', 'same entity reason'],
    'loc': ['multi-hop', 'same entity reason']
}
head_map = []
for editor_name in editor_names:
    head_map.append([])
    for sector in all_subjects:
        for discipline in all_subjects[sector]:
            vs = [v for model_name in ['gpt2-xl', 'gpt-j-6b', 'llama-3-8b'] for v in 
                get_values([discipline], editor_name, model_name, metric, 5, patterns_not[metric])]
            head_map[-1].append(np.mean(vs))
# plt.imshow(head_map)
# plt.show()
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

plt.rcParams['font.family'] = 'Times New Roman'

def draw_independent_heatbars(
        data, y_ticks, colormap, xticks=None, 
        aspect_ratio=(5, 25), fontsize=25, separator_color='white', 
        separator_linewidth=5, show_xticks=False, show_yticks=True, 
        save_path=None, save_formats=('svg',)):
    
    data = np.array(data, dtype=object)
    n_rows = len(data)
    n_cols = max(len(row) for row in data)

    img = np.ones((n_rows, n_cols, 4))  # RGBA
    text_positions = []

    for i, row in enumerate(data):
        cmap = plt.get_cmap(colormap)
        norm = mcolors.Normalize(vmin=min(row), vmax=max(row))
        for j, val in enumerate(row):
            rgba = cmap(norm(val))
            img[i, j, :] = rgba
            text_positions.append((j, i, val, rgba))

    fig_width = aspect_ratio[1] / aspect_ratio[0] * 5
    fig_height = aspect_ratio[0] / aspect_ratio[0] * 5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.imshow(img, aspect='auto')

    for j, i, val, rgba in text_positions:
        brightness = rgba[0]*0.299 + rgba[1]*0.587 + rgba[2]*0.114
        text_color = 'black' if brightness > 0.5 else 'white'
        ax.text(j, i+0.05, f'{val:.2f}' if val < 100 else f'{val:.1f}', ha='center', 
                va='center', color=text_color, fontsize=fontsize)

    for i in range(1, n_rows):
        ax.axhline(i - 0.5, color=separator_color, linewidth=separator_linewidth)

    if show_yticks:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(y_ticks, fontsize=fontsize, fontweight='bold')
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    if show_xticks and xticks:
        if len(xticks) != n_cols:
            raise ValueError("xticks length must match the number of columns in data.")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(xticks, rotation=0, ha='center', fontsize=fontsize, fontweight='bold')
    elif not show_xticks:
        ax.set_xticks([])
        ax.set_xticklabels([])
    else:
        ax.set_xticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        base, _ = os.path.splitext(save_path)
        for fmt in save_formats:
            plt.savefig(f"{base}.{fmt}", format=fmt, transparent=True)
    plt.show()

yticks = ['W/O', 'FT', 'IKE', 'ROME', 'SERAC', 'T-Patcher', 'GRACE', 'AlphaEdit'] 
xticks = [dis[:4].title() + ('.' if len(dis) > 4 else '') for sector in all_subjects for dis in all_subjects[sector]]
data = np.array(head_map)*100
colormaps = {'rel': 'Greens_r', 'gen': 'Blues_r', 'loc': 'Reds_r'}# Greens_r, Reds_r
show_xticks = {'rel': False, 'gen': False, 'loc': True}
draw_independent_heatbars(data, yticks, colormaps[metric], xticks, show_xticks=show_xticks[metric],
    save_path = f'figs/uniedit/performance_discipline/{metric}.svg')
