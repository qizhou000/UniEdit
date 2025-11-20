
# chemistry
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -enp "train-on-chemistry" -dn "UniEdit" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint"
# literature
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -enp "train-on-literature" -dn "UniEdit" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint"
# sociology
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -enp "train-on-sociology" -dn "UniEdit" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint"
# medicine
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -enp "train-on-medicine" -dn "UniEdit" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint"
# data science
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -enp "train-on-data science" -dn "UniEdit" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint"

