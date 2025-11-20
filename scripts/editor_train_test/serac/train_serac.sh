# serac gpt2-xl uniedit
python train_editor.py -et llm -en serac -mn gpt2-xl -dna UNIEDIT -bs 8 -dvc "cuda:0" -lkpt none -tnp UniEdit -eps 100 -sci 500
# serac gptj uniedit
python train_editor.py -et llm -en serac -mn gptj -dna UNIEDIT -bs 8 -dvc "cuda:0" -lkpt none -tnp UniEdit -eps 100 -sci 500
# serac llama-3 uniedit
python train_editor.py -et llm -en serac -mn llama-3 -dna UNIEDIT -bs 8 -dvc "cuda:0" -lkpt none -tnp UniEdit -eps 100 -sci 500
