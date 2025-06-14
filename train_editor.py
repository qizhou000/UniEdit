#%%
from utils import load_llm_editor
import os, argparse


def get_attr():
    def parse_lkpt(value:str):
        if value.lower() == 'none':
            return None
        return value
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument('-et', '--editor_type', type=str, help='Editor type: llm, vllm', required=True)
    parser.add_argument('-en', '--editor_name', type=str, help='Editor name: RECIPE, ...', required=True)
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: gpt2-xl, ...', required=True)
    parser.add_argument('-dna', '--data_name', type=str, help = 'Train dataset, including ZSRE, CF, RIPE.', required = True)
    parser.add_argument('-bs', '--batch_size', type=int, help = 'Train dataset sample number.', required = True)
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    # other settings
    parser.add_argument('-dn', '--data_n', type=int, default=None, help = 'Train dataset sample number.')
    parser.add_argument('-lkpt', '--load_ckpt_path', type=parse_lkpt, default = None, help='For Editors that needs training.')
    parser.add_argument('-edvc', '--extra_devices', type=int, nargs='+', default = [0], help='Extra CUDA devices, default empty.')
    parser.add_argument('-eps', '--epochs', type=int, default=1000, help = 'Train epochs.')
    parser.add_argument('-tnp', '--train_name_prefix', type=str, default=None, help = 'Train name prefix.')
    parser.add_argument('-sci', '--save_ckpt_per_i', type=int, default=1000, help = 'Save checkpoint per iteraions.')
    parser.add_argument('-lpi', '--log_per_i', type=int, default=1, help = 'Log per iteraions.')
    parser.add_argument('-ea', '--ema_alpha', type=float, default=0.1, help = 'EMA loss alpha.')
    parser.add_argument('-rs', '--random_seed', type=int, default=None, help = 'Random seed.')
    parser.add_argument('-dbs', '--data_buffer_size', type=int, default=4, help = 'Buffer size of data generator.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg = get_attr()
    if cfg.editor_type == 'llm':
        cfg.data_name = cfg.data_name.upper()
        # load editor
        editor = load_llm_editor(cfg.editor_name, cfg.edit_model_name, cfg.device, cfg.extra_devices, None, True)
        # load data
        if cfg.data_name == 'ZSRE':
            from dataset.llm import ZSRE
            data_path = os.path.join('data/meta-train/zsre/zsre_mend_train.json')
            train_data = ZSRE(data_path, cfg.data_n)
        elif cfg.data_name == 'CF':
            from dataset.llm import Counterfact
            data_path = os.path.join('data/meta-train/cf/counterfact-train.json')
            train_data = Counterfact(data_path, cfg.data_n)
        elif cfg.data_name == 'RIPE':
            from dataset.llm import RippleEffect
            data_path = os.path.join('data/meta-train/ripple_effect/ripe_train.json')
            train_data = RippleEffect(data_path, cfg.data_n)
        elif 'UNIEDIT' in cfg.data_name:
            from dataset.llm import UniEdit
            disciplines = [dis.lower() for dis in cfg.data_name.split('-') if dis != 'UNIEDIT']
            train_data = UniEdit('data/UniEdit/train', cfg.data_n, disciplines)
        # initialize and train
        editor.train_init(train_data, cfg.batch_size, train_name_prefix = cfg.train_name_prefix,
            load_ckpt_path = cfg.load_ckpt_path, save_ckpt_per_i = cfg.save_ckpt_per_i, 
            log_per_i = cfg.log_per_i, ema_alpha = cfg.ema_alpha, random_seed = cfg.random_seed,
            data_buffer_size = cfg.data_buffer_size) 
        editor.train(cfg.epochs)
    else: raise

