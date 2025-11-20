# Single Editing
# ike gpt2-xl UniEdit
python test_editor.py -et llm -en ike -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en ike -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en ike -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16
# ike gpt-j UniEdit
python test_editor.py -et llm -en ike -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en ike -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en ike -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16
# ike llama-3-8b UniEdit
python test_editor.py -et llm -en ike -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en ike -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en ike -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16
