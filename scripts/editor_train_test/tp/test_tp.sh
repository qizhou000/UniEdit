# Single Editing
# tp gpt2-xl UniEdit
python test_editor.py -et llm -en tp -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en tp -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16
# tp gpt-j UniEdit
python test_editor.py -et llm -en tp -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en tp -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16
# tp llama-3-8b UniEdit
python test_editor.py -et llm -en tp -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en tp -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16

# Sequential Editing
# tp gpt2-xl UniEdit
python test_editor.py -et llm -en tp -mn gpt2-xl -sen 10 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn gpt2-xl -sen 50 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn gpt2-xl -sen 100 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn gpt2-xl -sen 500 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn gpt2-xl -sen 1000 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
# tp gpt-j UniEdit
python test_editor.py -et llm -en tp -mn gpt-j -sen 10 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn gpt-j -sen 50 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn gpt-j -sen 100 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn gpt-j -sen 500 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn gpt-j -sen 1000 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
# tp llama-3-8b UniEdit
python test_editor.py -et llm -en tp -mn llama-3-8b -sen 10 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn llama-3-8b -sen 50 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn llama-3-8b -sen 100 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn llama-3-8b -sen 500 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000
python test_editor.py -et llm -en tp -mn llama-3-8b -sen 1000 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000


