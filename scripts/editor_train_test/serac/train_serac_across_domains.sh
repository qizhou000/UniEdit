
# serac gpt2-xl across domains
python train_editor.py -et llm -en serac -mn gpt2-xl -dna "UNIEDIT-Chemistry" -bs 8 -dvc "cuda:0" -lkpt none -tnp "UniEdit-Chemistry" -eps 20 -sci 500
python train_editor.py -et llm -en serac -mn gpt2-xl -dna "UNIEDIT-literature" -bs 8 -dvc "cuda:0" -lkpt none -tnp "UniEdit-literature" -eps 20 -sci 500
python train_editor.py -et llm -en serac -mn gpt2-xl -dna "UNIEDIT-sociology" -bs 8 -dvc "cuda:0" -lkpt none -tnp "UniEdit-sociology" -eps 20 -sci 500
python train_editor.py -et llm -en serac -mn gpt2-xl -dna "UNIEDIT-Medicine" -bs 8 -dvc "cuda:0" -lkpt none -tnp "UniEdit-Medicine" -eps 20 -sci 500
python train_editor.py -et llm -en serac -mn gpt2-xl -dna "UNIEDIT-data science" -bs 8 -dvc "cuda:0" -lkpt none -tnp "UniEdit-data" -eps 20 -sci 500
