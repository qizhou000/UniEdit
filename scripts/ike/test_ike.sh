# ike gpt2-xl UniEdit
# python test_editor.py -et llm -en ike -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-ike-gpt2-xl-UniEdit-1.log 2>&1 &
# ike gpt-j UniEdit
# python test_editor.py -et llm -en ike -mn gpt-j -sen 1 -dvc "cuda:1" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-ike-gpt-j-UniEdit-1.log 2>&1 &
# ike llama-3-3b UniEdit
# python test_editor.py -et llm -en ike -mn llama-3-3b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000  > jobs/nohup/test-ike-llama-3-3b-UniEdit-1.log 2>&1 &

# other seeds
# python test_editor.py -et llm -en ike -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8 # > jobs/nohup/test-ike-gpt2-xl-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ike -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8 # > jobs/nohup/test-ike-gpt-j-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ike -mn llama-3-3b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8 # > jobs/nohup/test-ike-llama-3-3b-UniEdit-1.log 2>&1 &

# python test_editor.py -et llm -en ike -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16 # > jobs/nohup/test-ike-gpt2-xl-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ike -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16 # > jobs/nohup/test-ike-gpt-j-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ike -mn llama-3-3b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16 # > jobs/nohup/test-ike-llama-3-3b-UniEdit-1.log 2>&1 &

