from editor.llm_editors import LLMBaseEditor, LLMBaseEditorWithTraining
from transformers.tokenization_utils_base import BatchEncoding
from editor.llms_for_edit import BaseLLMForEdit
from utils.GLOBAL import model_path_map, ROOT_PATH
from typing import Union, List, Dict, Optional
from torch import nn
import numpy as np
import torch, os


def print_list_structure(data, indent=0):
    indentation = '  ' * indent  
    if isinstance(data, list):
        print(f"List:")
        for index, item in enumerate(data):
            print(f"{indentation}  [{index}]:", end=' ')
            print_list_structure(item, indent + 1)  
    elif isinstance(data, tuple):
        print(f"Tuple:")
        for index, item in enumerate(data):
            print(f"{indentation}  ({index}):", end=' ')
            print_list_structure(item, indent + 1)  
    elif isinstance(data, torch.Tensor):
        print(f"Tensor: {data.shape}")
    else:
        print(f"{data}")

def find_module(module, module_path:str)->Union[torch.Tensor, nn.Module]:
    for comp in module_path.split('.'):
        if hasattr(module, comp):
            module = getattr(module, comp)
        elif comp.isdigit():
            module = module[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    return module
 
def move_to_device(data, device):
    '''Move list and dictionary nested PyTorch tensors to a specific device.'''
    if isinstance(data, (torch.Tensor, nn.Module, BatchEncoding)):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple([move_to_device(item, device) for item in data])
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (int, float, str, bool, type(None), np.integer, np.floating)):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def get_full_model_name(model_name_part:str)->str:
    model_name_part = model_name_part.lower()
    if 'blip2' in model_name_part:
        return 'blip2-opt-2.7b'
    elif 'llava' in model_name_part:
        return 'llava-v1.5-7b'
    elif 'mini' in model_name_part:
        if '4' in model_name_part and 'gpt' in model_name_part:
            return 'minigpt-4-vicuna-7b'
        else:
            raise
    elif 'bert' in model_name_part:
        if 'base' in model_name_part:
            if 'uncased' in model_name_part:
                return 'bert-base-uncased'
            elif 'cased' in model_name_part:
                return 'bert-base-cased'
            else:
                raise
        else:
            raise
    elif 'roberta' in model_name_part:
        return 'roberta-base'
    elif 'opt' in model_name_part:
        if '125m' in model_name_part:
            return 'opt-125m'
        else:
            raise
    elif 'gpt' in model_name_part:
        if 'j' in model_name_part:
            return 'gpt-j-6b'
        elif '2' in model_name_part:
            return 'gpt2-xl'
        else:
            raise
    elif 'llama' in model_name_part:
        if '2' in model_name_part:
            if 'chat' in model_name_part:
                return 'llama-2-7b-chat'
            else:
                return 'llama-2-7b'
        elif '3' in model_name_part:
            return 'llama-3-8b'
        elif '160m' in model_name_part:
            return 'llama-160m'
        else:
            raise
    raise

def get_editor_config_path(editor_name:str, edit_model_name:str, is_llm = True):
    return os.path.join(ROOT_PATH, 'configs', 'llms' if is_llm else 'vllms', 
            editor_name.lower(), '%s.yaml'%get_full_model_name(edit_model_name))

def get_model_path(model_name:str)->str:
    return model_path_map[get_full_model_name(model_name)]

################################################################################
################################ For LLMs ######################################
################################################################################
def load_llm_for_edit(model_name:str, device:str)->BaseLLMForEdit:
    model_name = get_full_model_name(model_name)
    model_path = get_model_path(model_name)
    print('Loading %s from "%s".'%(model_name, model_path))
    if 'gpt' in model_name and '2' in model_name and 'xl' in model_name :
        from editor.llms_for_edit.gpt2_xl.gpt2_xl import GPT2XLForEdit
        return GPT2XLForEdit(model_path, device)
    elif 'gpt' in model_name and 'j' in model_name:
        from editor.llms_for_edit.gpt_j.gpt_j import GPTJForEdit
        return GPTJForEdit(model_path, device)
    elif 'llama' in model_name and '3' in model_name:
        from editor.llms_for_edit.llama3.llama3 import LLAMA3ForEdit
        return LLAMA3ForEdit(model_path, device)


def load_llm_editor(editor_name:str, edit_model_name:str, device:int, 
        extra_devices:List[int] = [1], editor_ckpt_path = None, for_train = False):
    '''`for_train`: set features of some editors for training during initializing.'''
    editor_name = editor_name.lower()
    config_path = os.path.join(ROOT_PATH, 'configs', 'llms', editor_name.lower(), 
                               '%s.yaml'%get_full_model_name(edit_model_name))
    llm = load_llm_for_edit(edit_model_name, device)
    # load editor
    if editor_name == 'ft':
        from editor.llm_editors.ft.ft import FT, FTConfig
        config = FTConfig.from_yaml(config_path)
        editor = FT(llm, config, device) 
    elif editor_name == 'ike':
        from editor.llm_editors.ike.ike import IKE, IKEConfig
        config = IKEConfig.from_yaml(config_path)
        editor = IKE(llm, config, device) 
    elif editor_name == 'serac':
        from editor.llm_editors.serac.serac import SERAC, SERACConfig
        config = SERACConfig.from_yaml(config_path)
        editor = SERAC(llm, config, device = device)
    elif editor_name == 'rome':
        from editor.llm_editors.rome.rome import ROME, ROMEConfig
        config = ROMEConfig.from_yaml(config_path)
        editor = ROME(llm, config, 'data/rome-memit-stats', device = device)
    elif editor_name == 'tp':
        from editor.llm_editors.tp.tp import TP, TPConfig
        config = TPConfig.from_yaml(config_path)
        editor = TP(llm, config, device)
    elif editor_name == 'grace':
        from editor.llm_editors.grace.grace import GRACE, GRACEConfig
        config = GRACEConfig.from_yaml(config_path)
        editor = GRACE(llm, config, device)
    elif editor_name == 'alphaedit':
        from editor.llm_editors.alphaedit.alphaedit import AlphaEdit, AlphaEditConfig
        config = AlphaEditConfig.from_yaml(config_path)
        editor = AlphaEdit(llm, config, device, False)
    elif editor_name == 'woe':
        from editor.llm_editors.woe.woeditor import WOE, WOEConfig
        config = WOEConfig.from_yaml(config_path)
        editor = WOE(llm, config, device)
    else:
        raise BaseException('No such editor %s'%editor_name)
    if editor_ckpt_path != None and isinstance(editor, LLMBaseEditorWithTraining):
        editor.load_ckpt(editor_ckpt_path, True, for_train)
    return editor



