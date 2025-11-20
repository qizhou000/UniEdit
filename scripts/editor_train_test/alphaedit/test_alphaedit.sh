# Single Editing
# alphaedit gpt2-xl UniEdit
python test_editor.py -et llm -en alphaedit -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en alphaedit -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16
# alphaedit gpt-j UniEdit
python test_editor.py -et llm -en alphaedit -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en alphaedit -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16
# alphaedit llama-3-8b UniEdit
python test_editor.py -et llm -en alphaedit -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en alphaedit -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16

# Sequential Editing
# alphaedit gpt2-xl UniEdit
python test_editor.py -et llm -en alphaedit -mn gpt2-xl -sen 10 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn gpt2-xl -sen 50 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn gpt2-xl -sen 100 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn gpt2-xl -sen 500 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn gpt2-xl -sen 1000 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
# alphaedit gpt-j UniEdit
python test_editor.py -et llm -en alphaedit -mn gpt-j -sen 10 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn gpt-j -sen 50 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn gpt-j -sen 100 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn gpt-j -sen 500 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn gpt-j -sen 1000 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
# alphaedit llama-3-8b UniEdit
python test_editor.py -et llm -en alphaedit -mn llama-3-8b -sen 10 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn llama-3-8b -sen 50 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn llama-3-8b -sen 100 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn llama-3-8b -sen 500 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en alphaedit -mn llama-3-8b -sen 1000 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000


