#%%
from utils import get_full_model_name, load_llm_editor, load_vllm_editor
from evaluation.llm_editor_eval import LLMEditorEvaluation
from utils.GLOBAL import ROOT_PATH
import os, argparse, sys

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-et', '--editor_type', type=str, help='llm, vllm', required=True)
    parser.add_argument('-en', '--editor_name', type=str, help='Editor name: RECIPE, FT...', required=True)
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: GPT2-xl, GPT-J, LLAMA-2, ...', required=True)
    parser.add_argument('-sen', '--sequential_edit_n', type=int, help='Edit number.', required=True)
    parser.add_argument('-enp', '--eval_name_postfix', type=str, default = '', help='Postfix name of this evaluation.')
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    parser.add_argument('-ckpt', '--editor_ckpt_path', type=str, default = None, help='For Editors that needs training.')
    parser.add_argument('-dn', '--data_name', type=str, required = True, help = 'Evaluating dataset, including ZSRE, CF, RIPE, ....')
    parser.add_argument('-ds', '--data_settings', type=str, required = False, help = 'For extra dataset settngs.')
    parser.add_argument('-dsn', '--data_sample_n', type=int, default = None, help = 'Sample number for evaluation.')
    parser.add_argument('-rs', '--random_seed', type=int, default = -1, help = 'Random seed. -1: no random.')
    args = parser.parse_args()
    return args
 

if __name__ == '__main__':
    cfg = get_attr()
    cfg.editor_name = cfg.editor_name.lower()
    cfg.edit_model_name = get_full_model_name(cfg.edit_model_name)
    evaluation_name = cfg.data_name.upper()
    if cfg.eval_name_postfix != '':
        evaluation_name = '%s-%s'%(evaluation_name, cfg.eval_name_postfix)
    print(cfg)
    if cfg.editor_type == 'llm':
        editor = load_llm_editor(cfg.editor_name, cfg.edit_model_name, cfg.device, None, cfg.editor_ckpt_path, False)
        # load data
        if cfg.data_name == 'ZSRE':
            from dataset.llm import ZSRE
            data_path = 'data/evaluation/zsre/zsre_mend_eval.json'
            eval_data = ZSRE(data_path, cfg.data_sample_n, cfg.random_seed if cfg.random_seed != -1 else 1, False)
        elif cfg.data_name == 'CF':
            from dataset.llm import Counterfact
            data_path = 'data/evaluation/cf/counterfact-edit.json'
            eval_data = Counterfact(data_path, cfg.data_sample_n, cfg.random_seed if cfg.random_seed != -1 else 1, False) 
        elif cfg.data_name == 'RIPE':
            from dataset.llm import RippleEffect
            data_path = 'data/evaluation/ripple_effect/ripe_test.json'
            eval_data = RippleEffect(data_path, cfg.data_sample_n, cfg.random_seed if cfg.random_seed != -1 else 1, False)
        elif 'UniEdit' in cfg.data_name:
            from dataset.llm import UniEdit
            disciplines = [dis.lower() for dis in cfg.data_name.split('-') if dis != 'UniEdit']
            has_gen_patterns, has_loc_patterns = None, None
            if hasattr(cfg, 'data_settings') and cfg.data_settings != None:
                for d in cfg.data_settings.split('|'): # gen/gen_pattern_1~...|loc/loc_pattern_1~...
                    p = d.split('/')
                    if p[0] == 'gen': has_gen_patterns = p[1].split('~')
                    else: has_loc_patterns = p[1].split('~')
            print(f'disciplines = {disciplines}')
            print(f'has_loc_patterns = {has_loc_patterns}')
            print(f'has_gen_patterns = {has_gen_patterns}')
            eval_data = UniEdit('data/UniEdit/test', cfg.data_sample_n, disciplines, 
                has_gen_patterns=has_gen_patterns, has_loc_patterns=has_loc_patterns,
                random_seed = cfg.random_seed if cfg.random_seed != -1 else 1, add_wiki_loc = False)
        ev = LLMEditorEvaluation(editor, eval_data, evaluation_name, 'eval_results')
    else: raise
    # evaluate
    if cfg.random_seed == -1:
        ev.evaluate_sequential_edit(cfg.sequential_edit_n, False, None) 
    else:
        ev.evaluate_sequential_edit(cfg.sequential_edit_n, True, cfg.random_seed)
