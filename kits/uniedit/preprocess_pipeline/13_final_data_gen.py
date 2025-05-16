#%%
from kits.uniedit.tools.final_edit_data_generate import FinalEditDataGenerator
from datetime import datetime
from time import time
import argparse

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sb', '--subject', type=str, required=True)
    args = parser.parse_args()
    return args
class cfg:
    subject = 'biology'
cfg = get_attr()
edg = FinalEditDataGenerator(ai_gen_temperature = 0.5)
edg.generate_all_final_data(cfg.subject, max_data_gen = 15000, 
    save_every = 5, max_thread = 5, random_seed = 1234,
    proc_start_time = datetime.strptime('00:45', '%H:%M'),
    proc_end_time = datetime.strptime('08:15', '%H:%M'),
    # proc_end_time = datetime.strptime('23:15', '%H:%M'),
)
