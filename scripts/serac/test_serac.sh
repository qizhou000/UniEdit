# serac gpt2-xl UniEdit
# python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-gpt2-xl-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt2-xl -sen 10 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 1000 > jobs/nohup/test-serac-gpt2-xl-UniEdit-10.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt2-xl -sen 50 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 1500 > jobs/nohup/test-serac-gpt2-xl-UniEdit-50.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt2-xl -sen 100 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 2000 > jobs/nohup/test-serac-gpt2-xl-UniEdit-100.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt2-xl -sen 500 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 2500 > jobs/nohup/test-serac-gpt2-xl-UniEdit-500.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1000 -dvc "cuda:0" -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-gpt2-xl-UniEdit-1000.log 2>&1 &
# serac gpt-j UniEdit
# python test_editor.py -et llm -en serac -mn gpt-j -sen 1 -dvc "cuda:1" -ckpt "records/serac/gpt-j-6b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-gpt-j-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt-j -sen 10 -dvc "cuda:1" -ckpt "records/serac/gpt-j-6b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 1000 > jobs/nohup/test-serac-gpt-j-UniEdit-10.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt-j -sen 50 -dvc "cuda:1" -ckpt "records/serac/gpt-j-6b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 1500 > jobs/nohup/test-serac-gpt-j-UniEdit-50.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt-j -sen 100 -dvc "cuda:0" -ckpt "records/serac/gpt-j-6b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 2000 > jobs/nohup/test-serac-gpt-j-UniEdit-100.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt-j -sen 500 -dvc "cuda:1" -ckpt "records/serac/gpt-j-6b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 2500 > jobs/nohup/test-serac-gpt-j-UniEdit-500.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt-j -sen 1000 -dvc "cuda:1" -ckpt "records/serac/gpt-j-6b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-gpt-j-UniEdit-1000.log 2>&1 &
# serac llama-3-8b UniEdit
# python test_editor.py -et llm -en serac -mn llama-3-8b -sen 1 -dvc "cuda:2" -ckpt "records/serac/llama-3-8b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-llama-3-8b-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en serac -mn llama-3-8b -sen 10 -dvc "cuda:2" -ckpt "records/serac/llama-3-8b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 1000 > jobs/nohup/test-serac-llama-3-8b-UniEdit-10.log 2>&1 &
# python test_editor.py -et llm -en serac -mn llama-3-8b -sen 50 -dvc "cuda:0" -ckpt "records/serac/llama-3-8b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 1500 > jobs/nohup/test-serac-llama-3-8b-UniEdit-50.log 2>&1 &
# python test_editor.py -et llm -en serac -mn llama-3-8b -sen 100 -dvc "cuda:2" -ckpt "records/serac/llama-3-8b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 2000 > jobs/nohup/test-serac-llama-3-8b-UniEdit-100.log 2>&1 &
# python test_editor.py -et llm -en serac -mn llama-3-8b -sen 500 -dvc "cuda:0" -ckpt "records/serac/llama-3-8b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 2500 > jobs/nohup/test-serac-llama-3-8b-UniEdit-500.log 2>&1 &
# python test_editor.py -et llm -en serac -mn llama-3-8b -sen 1000 -dvc "cuda:2" -ckpt "records/serac/llama-3-8b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-llama-3-8b-UniEdit-1000.log 2>&1 &




# other random seed
# python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -rs 8 -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-gpt2-xl-UniEdit-1-rs8.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt2-xl -sen 1 -dvc "cuda:0" -rs 16 -ckpt "records/serac/gpt2-xl/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-gpt2-xl-UniEdit-1-rs16.log 2>&1 &

# python test_editor.py -et llm -en serac -mn gpt-j -sen 1 -dvc "cuda:0" -rs 8 -ckpt "records/serac/gpt-j-6b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-gpt-j-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en serac -mn gpt-j -sen 1 -dvc "cuda:0" -rs 16 -ckpt "records/serac/gpt-j-6b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-gpt-j-UniEdit-1.log 2>&1 &

# python test_editor.py -et llm -en serac -mn llama-3-8b -sen 1 -dvc "cuda:3" -rs 8 -ckpt "records/serac/llama-3-8b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-llama-3-8b-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en serac -mn llama-3-8b -sen 1 -dvc "cuda:3" -rs 16 -ckpt "records/serac/llama-3-8b/UniEdit/checkpoints/Checkpoint" -dn UniEdit -dsn 5000 > jobs/nohup/test-serac-llama-3-8b-UniEdit-1.log 2>&1 &
