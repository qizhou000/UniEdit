from editor.llm_editors import LLMBaseEditor
from editor.llms_for_edit import BaseLLMForEdit
from dataset.llm import BaseLLMEditData
from typing import List, Dict, Union
from collections import defaultdict
from . import BaseEditorEvaluation
from datetime import datetime
from copy import deepcopy
import torch, os, json
from tqdm import tqdm
from time import time
import numpy as np

class LLMEditorEvaluation(BaseEditorEvaluation):
    def __init__(self, editor:LLMBaseEditor, eval_data:BaseLLMEditData, 
        evaluation_name = None, results_dir = 'eval_results') -> None:
        '''`results_dir` & `evaluation_name`: Used to create result directory.
            `evaluation_name` can be set as dataset name.'''
        super().__init__()
        self.editor = editor
        self.eval_data = eval_data
        editor_name, model_name = editor.name_of_editor_and_model()
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        evaluation_name = evaluation_name if evaluation_name else t
        self.result_dir = os.path.join(results_dir, 'llms', editor_name, model_name, evaluation_name)
        print('Evaluation results directory: ', self.result_dir)


    def evaluate_single_edit(self):
        raise "Use evaluate_sequential_edit(1)"

    def evaluate_sequential_edit(self, edit_n = 10, random = False, seed = None, target_check_topk = 5):
        # preprocess data for sequential editing evaluation
        def split_data(data): 
            splited_data = []
            splited_data_ns = []
            now_split = []
            now_split_edit_n = 0
            for d in data:
                now_split.append(d)
                now_split_edit_n += len(d['requests'])
                if now_split_edit_n >= edit_n:
                    splited_data.append(now_split)
                    splited_data_ns.append(now_split_edit_n)
                    now_split = []
                    now_split_edit_n = 0
            return splited_data, splited_data_ns
        data_indexs = list(range(self.eval_data.data_n()))
        seed = seed if seed != None else np.random.randint(1, 999999)
        if random: 
            np.random.default_rng(seed).shuffle(data_indexs)
        eval_data = deepcopy(self.eval_data.get_data_by_ids_without_loc_edit_collision(data_indexs))
        if eval_data == None:
            eval_data = deepcopy(self.eval_data.get_data_by_ids(data_indexs))
        result_data = deepcopy(eval_data)
        eval_data, eval_data_ns = split_data(eval_data)
        result_data, _ = split_data(result_data)
        # evaluate
        editor = self.editor
        tokenizer = editor.llm.tokenizer
        print('Evaluating for %s on %s with %s sequential editing on %s.'
            %(*editor.name_of_editor_and_model(), edit_n, self.eval_data.dataset_name()))
        results = [] 
        editor.restore_to_original_model() 
        for split_rd, split_ed in zip(tqdm(result_data, 'Evaluating', ncols = 60), eval_data):
            split_res = []
            for rd, ed in zip(tqdm(split_rd, 'Preparing', leave = False, ncols = 60), split_ed):
                rd['reliability'] = rd.pop('requests') 
                for r in rd['reliability']:
                    r['target'] = r.pop('target_new') 
                for loc_name in ed['locality'].keys(): # predict before edit for locality data
                    for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                        input_ids, label_ids, label_masks = editor.llm.prompts_targets_to_xym(
                            [edl['prompt']], [edl['target']])
                        logits = editor.llm.get_llm_outpt(input_ids).logits
                        before_edit_ids = torch.softmax(logits, -1).argmax(-1)[:, -label_ids.shape[1]:] # [1, l2]
                        rdl['predict_before_edit'] = tokenizer.decode(before_edit_ids[label_masks.to(bool)])
                        edl['before_edit_ids'] = before_edit_ids[label_masks.to(bool)]
            for rd, ed in zip(tqdm(split_rd, 'Editing', leave = False, ncols = 60), split_ed): # edit 
                for rdr, edr in zip(rd['reliability'], ed['requests']):
                    start_t = time()
                    editor.edit_one_piece(edr)
                    rdr['edit_time'] = time() - start_t
            editor.save_current_edit_status()
            for rd, ed in zip(tqdm(split_rd, 'Testing', leave = False, ncols = 60), split_ed): # compute scores 
                rd = self.__test_after_edit__(ed, rd, target_check_topk)
                split_res.append(rd)
            editor.restore_to_original_model()
            results.append(split_res)
        # save results
        save_dir = os.path.join(self.result_dir, 'sequential_edit_%s'%edit_n)
        self.save_results(os.path.join(save_dir, '%sresults.json'%('seed_%s_'%seed if random else '')), results)
        split_mean = [self.get_mean_results(sr) for sr in results]
        for mr, n in zip(split_mean, eval_data_ns):
            mr['sequential_edit_n'] = n
        total_mean = self.get_mean_results([r for sr in results for r in sr])
        total_mean['total_edit_n'] = sum(eval_data_ns)
        mean_results = {"total_mean": total_mean, "split_mean": split_mean}
        self.save_results(os.path.join(save_dir, '%smean_results.json'%('seed_%s_'%seed if random else '')), mean_results)
        return results

    def __test_after_edit__(self, edit_data, result_data, target_check_topk = 3):
        editor = self.editor
        llm = self.editor.llm
        tokenizer = llm.tokenizer
        def accuracy_and_prediction(input_ids, label_ids, label_masks):
            # label_ids/label_masks: [1, l2]
            assert len(label_ids) == 1 and len(label_masks) == 1
            logits = llm.get_llm_outpt(input_ids).logits # [1,l1,d]
            pre_y = torch.softmax(logits, -1).argmax(-1) # [1, l1]
            pre_y = pre_y[:, -label_ids.shape[1]:] # [1, l2]
            acc = ((pre_y == label_ids) * label_masks).sum()/label_masks.sum() 
            return float(acc), pre_y
        def check_target_in_topk_prob_distribution(prompt:str, target:str, topk:int):
            # check if the tokens of target are in the topk prob distribution
            input_ids, label_ids, label_masks = llm.prompts_targets_to_xym([prompt], [target])
            label_masks = label_masks[0].to(bool) # [answer_length + 1]
            logit = llm.get_llm_outpt(input_ids).logits[0, -len(label_masks):][label_masks]
            prob = torch.softmax(logit, 1) # [answer_length, vocabulary_length]
            topk_tokens = torch.topk(prob, topk, 1).indices # [answer_length, topk]
            topk_pre_strs = tokenizer.batch_decode(topk_tokens.reshape(-1, 1))
            topk_pre_strs = [topk_pre_strs[i:i+topk] for i in range(0, len(topk_pre_strs), topk)]
            each_in = []
            label_ids = label_ids[0, label_masks] # [answer_length]
            for t, tt in zip(label_ids, topk_tokens):
                each_in.append(t in tt)
            all_in = all(each_in)
            label_ids = label_ids.reshape(-1, 1).tolist()
            target_tokens = tokenizer.batch_decode(label_ids)
            return all_in, each_in, topk_pre_strs, target_tokens
        # reliability
        for rdr, edr in zip(result_data['reliability'], edit_data['requests']):
            input_ids, label_ids, label_masks = llm.prompts_targets_to_xym(
                                        [edr['prompt']], [edr['target_new']])
            acc, pre_y = accuracy_and_prediction(input_ids, label_ids, label_masks)
            rdr['predict_after_edit'] = tokenizer.decode(pre_y[label_masks.to(bool)])
            rdr['acc'] = acc
        # generality
        for gen_name in edit_data['generality']:
            for rdg, edg in zip(result_data['generality'][gen_name], edit_data['generality'][gen_name]):
                extra_edit = False
                if 'related_hops' in edg.keys():
                    for erh, rrh in zip(edg['related_hops'], rdg['related_hops']):
                        all_in, _, _, _ = check_target_in_topk_prob_distribution(
                            erh['prompt'], erh['target'], target_check_topk)
                        rrh['edited'] = not all_in
                        if not all_in:
                            extra_edit = True
                            erh['target_new'] = erh.pop('target')
                            editor.edit_one_piece(erh)
                tct = edg['target_check_topk'] if 'target_check_topk' in edg.keys() else target_check_topk
                all_in, each_in, topk_pre_strs, target_tokens = check_target_in_topk_prob_distribution(
                    edg['prompt'], edg['target'], tct)
                rdg['target_check_topk'] = tct
                rdg['post_edit_predict_tokens'] = topk_pre_strs
                rdg['target_tokens'] = target_tokens
                rdg['acc'] = sum(each_in) / len(each_in) # all_in
                if extra_edit:
                    editor.restore_to_saved_edit_status()
        # locality
        for loc_name in edit_data['locality']:
            for rdl, edl in zip(result_data['locality'][loc_name], edit_data['locality'][loc_name]):
                tct = edl['target_check_topk'] if 'target_check_topk' in edl.keys() else target_check_topk
                all_in, each_in, topk_pre_strs, _ = check_target_in_topk_prob_distribution(
                    edl['prompt'], edl['target'], tct)
                rdl['target_check_topk'] = tct
                rdl['post_edit_predict_tokens'] = topk_pre_strs
                rdl['pre_edit_predict_tokens'] = tokenizer.batch_decode(
                    edl['before_edit_ids'].reshape(-1, 1))
                rdl['acc'] = sum(tt in tps for tps, tt in zip(
                    topk_pre_strs, rdl['pre_edit_predict_tokens'])
                    )/len(rdl['pre_edit_predict_tokens'])
                # all(tt in pept for pept, tt in 
                    # zip(rdl['post_edit_predict_tokens'], rdl['pre_edit_predict_tokens'])) 
        return result_data

    def get_mean_results(self, results:List[Dict]):
        """Get numbers from a result: {
            "reliability": [
                {"acc": float, "edit_time": float}, 
                {"acc": float, "edit_time": float}, ...]
            "generality": {
                sub_metric_1: [{"acc": float}, {"acc": float}, ...], 
                sub_metric_2: [{"acc": float}, {"acc": float}, ...], ...}
            "locality": {
                sub_metric_1: [{"acc": float}, {"acc": float}, ...], 
                sub_metric_2: [{"acc": float}, {"acc": float}, ...], ...}
        }
        """
        mean_res = {"reliability": {}, "generality": {}, "locality": {}}
        # sum values
        for r in results:
            for rr in r['reliability']:
                for value_name, value in rr.items():
                    if isinstance(value, (int, float)):
                        if value_name not in mean_res['reliability']:
                            mean_res['reliability'][value_name] = [0, 0]
                        mean_res['reliability'][value_name][0] += value
                        mean_res['reliability'][value_name][1] += 1
            for sub_metric in r['generality'].keys():
                if sub_metric not in mean_res['generality']:
                    mean_res['generality'][sub_metric] = {}
                for sub_res in r['generality'][sub_metric]:
                    for value_name, value in sub_res.items():
                        if isinstance(value, (int, float)):
                            if value_name not in mean_res['generality'][sub_metric]:
                                mean_res['generality'][sub_metric][value_name] = [0, 0]
                            mean_res['generality'][sub_metric][value_name][0] += value
                            mean_res['generality'][sub_metric][value_name][1] += 1
            for sub_metric in r['locality'].keys():
                if sub_metric not in mean_res['locality']:
                    mean_res['locality'][sub_metric] = {}
                for sub_res in r['locality'][sub_metric]:
                    for value_name, value in sub_res.items():
                        if isinstance(value, (int, float)):
                            if value_name not in mean_res['locality'][sub_metric]:
                                mean_res['locality'][sub_metric][value_name] = [0, 0]
                            mean_res['locality'][sub_metric][value_name][0] += value
                            mean_res['locality'][sub_metric][value_name][1] += 1
        # compute mean results
        for value_name, value in mean_res['reliability'].items():
            mean_res['reliability'][value_name] = value[0] / value[1]
        mean_acc, acc_n = 0, 0
        for sub_metric in mean_res['generality'].keys():
            for value_name, value in mean_res['generality'][sub_metric].items():
                mean_res['generality'][sub_metric][value_name] = value[0] / value[1]
                if value_name == 'acc':
                    mean_acc += mean_res['generality'][sub_metric][value_name]
                    acc_n += 1
        mean_res['generality']['mean_acc'] = mean_acc / acc_n if acc_n != 0 else 0
        mean_acc, acc_n = 0, 0
        for sub_metric in mean_res['locality'].keys():
            for value_name, value in mean_res['locality'][sub_metric].items():
                mean_res['locality'][sub_metric][value_name] = value[0] / value[1]
                if value_name == 'acc':
                    mean_acc += mean_res['locality'][sub_metric][value_name]
                    acc_n += 1
        mean_res['locality']['mean_acc'] = mean_acc / acc_n if acc_n != 0 else 0
        return mean_res

    def save_results(self, save_path:str, results:Dict, decimal = 4):
        def set_decimal(r):
            if isinstance(r, list):
                for i in range(len(r)):
                    r[i] = set_decimal(r[i])
            elif isinstance(r, dict) or isinstance(r, defaultdict):
                for k in r.keys():
                    r[k] = set_decimal(r[k])
            elif isinstance(r, float):
                r = round(r, decimal)
            return r
        res = deepcopy(results)
        res = set_decimal(res)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(os.path.join(save_path), 'w') as f:
            json.dump(res, f, indent = 4)

