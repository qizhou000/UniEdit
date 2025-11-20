# Single Editing
# serac gpt2-xl UniEdit
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 -rs 16
# serac gpt-j UniEdit
python test_editor.py -et llm -en serac -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en serac -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 -rs 16
# serac llama-3-8b UniEdit
python test_editor.py -et llm -en serac -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 -rs 8
python test_editor.py -et llm -en serac -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 -rs 16

# Sequential Editing
# serac gpt2-xl UniEdit
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 10 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 50 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 100 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 500 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1000 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
# serac gpt-j UniEdit
python test_editor.py -et llm -en serac -mn gpt-j -sen 10 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn gpt-j -sen 50 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn gpt-j -sen 100 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn gpt-j -sen 500 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn gpt-j -sen 1000 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
# serac llama-3-8b UniEdit
python test_editor.py -et llm -en serac -mn llama-3-8b -sen 10 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn llama-3-8b -sen 50 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn llama-3-8b -sen 100 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn llama-3-8b -sen 500 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000
python test_editor.py -et llm -en serac -mn llama-3-8b -sen 1000 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000

