from typing import Dict, List, Tuple, Optional, Union
from numpy.random._generator import Generator as RNG
from torch.utils.tensorboard import SummaryWriter 
from ..llms_for_edit import BaseLLMForEdit
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, asdict
from dataset.llm import BaseLLMEditData
from abc import ABC, abstractmethod
from dataset import ParallelDataset
from datetime import datetime
from .. import BaseConfig
from copy import deepcopy
import json, yaml, os, torch
from tqdm import tqdm
from torch import nn
import numpy as np
        
############################################################################
######################## LLM Base Editor Classes ########################### 
############################################################################
class LLMBaseEditor(ABC):
    def __init__(self, llm:BaseLLMForEdit, device:str):
        if not isinstance(llm, BaseLLMForEdit): raise
        self.llm = llm
        self.llm.set_device(device)
        self.device = 'cuda:0' if device == 'auto' else device
        assert self.if_model_decoder_only() # temporary only support decoder-only llm

    def if_model_decoder_only(self)->bool:
        if self.llm.model.config.is_encoder_decoder:
            return False
        return True

    @abstractmethod
    def name_of_editor_and_model(self)->Tuple[str, str]:
        '''return editor_name:str, model_name:str '''

    @abstractmethod
    def restore_to_original_model(self):
        '''A method for restoring the original model weights after editing with as 
        low GPU memory usage as possible. '''

    @abstractmethod
    def edit_one_piece(self, request:Dict):
        '''request = {'image': PILImage, 'prompt': str, 'target_new': str, ...}'''

    @abstractmethod
    def edit_batch(self, requests:List[Dict]):
        '''Assume: 
        requests = [
          {'image': PILImage, 'prompt': str, 'target_new': str, ...},
          {'image': PILImage, 'prompt': str, 'target_new': str, ...}, ...
        ]
        '''

    @abstractmethod
    def if_can_batch_edit(self)->bool:
        pass

    @abstractmethod
    def save_current_edit_status(self):
        pass

    @abstractmethod
    def restore_to_saved_edit_status(self):
        pass



class LLMBaseEditorWithTraining(LLMBaseEditor):
    def __init__(self, llm:BaseLLMForEdit, config:BaseConfig, device:str):
        super().__init__(llm, device)
        self.cfg = config 

    @abstractmethod
    def get_modules_for_training(self)->Dict[str, nn.Module]:
        '''
        Get modules for training, used for `self.save_ckpt` and `self.load_ckpt`.
        Assume return `train_modules`: Dict[str, nn.Module]
        '''
    
    @abstractmethod
    def reinit_train_parameters(self):
        '''Reinitialize parameters of modules to be trained.'''

    @abstractmethod
    def preprocess_train_data(self, llm_edit_data:BaseLLMEditData)->Union[List, BaseLLMEditData]:
        '''Process raw training data, used in `self.train_init`. 
        Return the data list or `BaseLLMEditData` used to sample batch data every iteration.'''
        
    @abstractmethod
    def organize_batch_data(self, a_batch_of_training_data:List):
        '''
        This function is used to dynamically organized data during training in 
            `self.data_generator`.
        `a_batch_of_training_data`: a batch/list of training data.
        return `a_batch_of_organized_training_data`
        '''
        
    @abstractmethod
    def train_a_batch(self, a_batch_of_organized_training_data):
        '''
        This function would be inputted a batch of organized training data and 
        should train the editor once. 
        `a_batch_of_organized_training_data`: a batch of organized training data 
            coming from `self.organize_batch_data`.
        return loss:float, log_dict: Dict[str, int]
        '''
    
    @abstractmethod
    def get_a_new_optimizer(self)->Union[torch.optim.Optimizer, 
        Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]]:
        ''' Initialize optimizer for training.
            Assume return an `opt`: 
                torch.optim.Optimizer
            or an `opt` and its learning rate scheduler: 
                [torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler].
        '''

    @abstractmethod
    def set_train(self, is_train:bool):
        '''Set training state for editor. '''

    def rand_trans_s(self, s: str, rng: RNG, key_words: List[str], 
            noise_words: List[str], prefixes: List[str] = [], postfixes: List[str] = [],
            max_remove_rate: float = 0.5, max_permut_rate: float = 0.5, 
            max_noise_rate: float = 0.5):
        '''Randomly transform sentence: 
        1. Remove tokens; 2. Rermute tokens; 3. Add noise tokens.'''
        split_s = s.split(' ')
        split_s_n = len(split_s)
        # permutation
        max_permut_n = split_s_n * max_permut_rate
        if max_permut_n > 1:
            trans_idxs = list(range(split_s_n))
            rng.shuffle(trans_idxs)
            trans_idxs = trans_idxs[:rng.integers(1, max_permut_n + 1)]
            trans_split_s = [split_s[i] for i in trans_idxs]
            rng.shuffle(trans_split_s)
            for i, w in zip(trans_idxs, trans_split_s):
                split_s[i] = w
        # remove
        max_remove_n = split_s_n * max_remove_rate
        if max_remove_n > 0:
            remove_n = rng.integers(0, max_remove_n + 1)
            removed_idxs = list(range(split_s_n))
            rng.shuffle(removed_idxs)
            removed_idxs = removed_idxs[:remove_n]
            split_s = [w for i, w in enumerate(split_s) 
                    if i not in removed_idxs or w in key_words]
        # noise
        max_noise_n = split_s_n * max_noise_rate
        if len(noise_words) > 0 and max_noise_n > 0: 
            for i in range(rng.integers(0, max_noise_n + 1)):
                split_s.insert(rng.integers(0, split_s_n + 1), noise_words[rng.integers(0, len(noise_words))])
        # pre-post-fix
        s = ' '.join(split_s)
        if len(prefixes) != 0:
            ps = prefixes[rng.integers(len(prefixes))]
            if ps[-1] != ' ' and s[0] != ' ':
                s = ps + ' ' + s 
            else:
                s = ps + s
        if len(postfixes) != 0:
            ps = postfixes[rng.integers(len(postfixes))]
            if ps[0] != ' ' and s[-1] != ' ':
                s = s + ' ' + ps
            else:
                s = s + ps
        return s

    def set_random_seeds(self, seed:int):
        import random, time
        from torch.backends import cudnn
        if seed == None:
            seed = int(time.time()*10000)%99999999
        print('Random seed is', seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        np.random.seed(seed)  
        random.seed(seed)   
        cudnn.benchmark = False # Do not test and select a optim algorithm for CNN
        cudnn.deterministic = True # Deterministic mode for Convolution/Pooling
        self.random_seed = seed

    def other_train_init_final(self):
        '''Called at the end of `self.train_init`.'''

    def other_train_init_begin(self):
        '''Called at the begin of `self.train_init`.'''

    def train_init(self, llm_edit_data:BaseLLMEditData, batch_size:int, 
            records_dir:str = 'records', train_name_prefix:str = None, 
            train_name:str = None, load_ckpt_path:str = None, 
            save_ckpt_per_i:int = 3000, log_per_i:int = 10, 
            ema_alpha:float = 0.1, random_seed:int = None, 
            data_buffer_size = 8, seed_init_train_params_if_no_ckpt_path = True):  
        '''Used to initialize data generator `self.data_generator`, checkpoint/log 
            directory, writer, and optimizer. '''
        self.set_random_seeds(random_seed)
        self.other_train_init_begin()
        # initialize data generator
        assert isinstance(llm_edit_data, BaseLLMEditData)
        training_data = self.preprocess_train_data(llm_edit_data)
        if isinstance(training_data, BaseLLMEditData):
            data_n = training_data.data_n()
            def get_data_by_ids_func(ids):
                return self.organize_batch_data(training_data.get_data_by_ids(ids))
        elif isinstance(training_data, list):
            data_n = len(training_data)
            def get_data_by_ids_func(ids):
                return self.organize_batch_data([training_data[i] for i in ids])
        else: raise 
        self.data_generator = ParallelDataset(data_n, get_data_by_ids_func, 
            batch_size, True, data_buffer_size, False, self.random_seed, True)
        # initialize checkpoint/log directory and writer
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        train_name = (train_name_prefix + '-' if train_name_prefix else "") + \
            (train_name if train_name else t)
        records_dir = os.path.join(records_dir, *self.name_of_editor_and_model(), train_name)
        self.save_ckpt_dir = os.path.join(records_dir, 'checkpoints')
        os.makedirs(self.save_ckpt_dir, exist_ok = True)
        logs_path = os.path.join(records_dir, 'logs')
        os.makedirs(logs_path, exist_ok = True)
        with open(os.path.join(records_dir, 'config.yaml'), 'w') as f:
            cfg = asdict(deepcopy(self.cfg))
            cfg['train_batch_size'] = batch_size
            cfg['random_seed'] = self.random_seed
            cfg['sample_count'] = data_n
            print("%s configs:"%self.name_of_editor_and_model()[0], cfg)
            yaml.dump(cfg, f)
        self.log_writer = SummaryWriter(logs_path)
        self.save_ckpt_per_i = save_ckpt_per_i
        self.log_per_i = log_per_i
        self.ema_alpha = ema_alpha
        # initialize optimizer and load checkpoints
        opt = self.get_a_new_optimizer()
        self.opt, self.lr_scheduler = [opt, None] if isinstance(opt, torch.optim.Optimizer) else opt
        if load_ckpt_path:
            assert os.path.isfile(load_ckpt_path)
            self.train_i, self.train_epoch, _, self.ema_loss = self.load_ckpt(load_ckpt_path, True)
            self.ema_loss = 1 if self.ema_loss == None else self.ema_loss
        else:
            if seed_init_train_params_if_no_ckpt_path:
                print('Train parameters are reinitialized with seed %s.'%self.random_seed)
                self.reinit_train_parameters()
            self.train_i = self.train_epoch = self.ema_loss = 1
        # initialize other settings
        self.other_train_init_final()


    def train(self, total_epochs):
        if self.log_writer == None:
            raise "Call `self.train_init()` to initialize training first!"
        print('Checkpoints dir: ', self.save_ckpt_dir)
        start_epoch = self.train_epoch
        self.set_train(True) 
        for self.train_epoch in range(start_epoch, total_epochs + 1): 
            progress_bar = tqdm(total = self.data_generator.sample_count, 
                position = 0, leave = True, desc = "Epoch %d"%self.train_epoch, dynamic_ncols = True)
            for a_batch_samples, samp_n in self.data_generator:
                # train after edit
                loss, log_dict = self.train_a_batch(a_batch_samples)
                self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss
                # log
                log_dict['Loss'] = loss
                log_dict['EMA Loss'] = self.ema_loss
                log_dict['Epoch'] = self.train_epoch
                if self.train_i % self.log_per_i == 0:
                    self.write_logs(self.train_i, log_dict)
                if self.train_i % self.save_ckpt_per_i == 0:
                    self.save_ckpt(self.train_i, self.train_epoch, loss, self.ema_loss)
                self.train_i += 1 
                progress_bar.update(samp_n)
            progress_bar.close() 
        self.set_train(False)

    def write_logs(self, i, logs:dict):
        for log_name, log in logs.items():
            if type(log) == dict:
                logs1 = {}
                for n, l in log.items():
                    logs1[log_name + '-' + n] = l
                self.write_logs(i, logs1)
            else:   
                self.log_writer.add_scalar(log_name, log, i)

    # def other_objects_save_to_ckpt(self): 
    #     '''Return the objects should be save into checkpoint, except to training modules.'''

    def save_ckpt(self, i:int, epoch:int, loss:float, ema_loss:float = None):
        '''Save checkpoint.'''
        train_modules = self.get_modules_for_training()
        ckpt = {
            'i': i,
            'epoch': epoch,
            'loss': loss,
            'ema_loss': ema_loss,
            'train_modules': {k:v.state_dict() for k, v in train_modules.items()},
            'opt': self.opt.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler != None else None
        }
        if ema_loss != None:
            ckpt_name = 'epoch-%d-i-%d-ema_loss-%.4f'%(int(epoch), int(i), float(ema_loss))
        else:
            ckpt_name = 'epoch-%d-i-%d-loss-%.4f'%(int(epoch), int(i), float(loss))
        ckpt_path = os.path.join(self.save_ckpt_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, ckpt_path, restrict = True, load_opt = True):
        '''Load checkpoint.'''
        ckpt = torch.load(ckpt_path, 'cpu', weights_only = True)
        train_modules = self.get_modules_for_training()
        for k in train_modules.keys():
            if isinstance(train_modules[k], nn.Module):
                train_modules[k].load_state_dict(ckpt['train_modules'][k], restrict)
            elif isinstance(train_modules[k], torch.optim.Optimizer):
                train_modules[k].load_state_dict(ckpt['train_modules'][k])
            else:
                raise
        if load_opt:
            self.opt.load_state_dict(ckpt['opt'])
            if self.lr_scheduler != None and ckpt['lr_scheduler'] != None:
                self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        print('Load %s checkpoint from %s.'%(self.name_of_editor_and_model()[0], ckpt_path))
        return ckpt['i'], ckpt['epoch'], ckpt['loss'], ckpt['ema_loss']



