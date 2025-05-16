# woe gpt2-xl UniEdit
# python test_editor.py -et llm -en woe -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-woe-gpt2-xl-UniEdit-1.log 2>&1 &
# woe gpt-j UniEdit
# python test_editor.py -et llm -en woe -mn gpt-j -sen 1 -dvc "cuda:1" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-woe-gpt-j-UniEdit-1.log 2>&1 &
# woe llama-3-3b UniEdit
# python test_editor.py -et llm -en woe -mn llama-3-3b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-woe-llama-3-3b-UniEdit-1.log 2>&1 &


# other seed
# python test_editor.py -et llm -en woe -mn gpt2-xl -sen 1 -dvc "cuda:1" -ckpt none -dn UniEdit -dsn 5000 -rs 8  > jobs/nohup/test-woe-gpt2-xl-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en woe -mn gpt-j -sen 1 -dvc "cuda:2" -ckpt none -dn UniEdit -dsn 5000 -rs 8  > jobs/nohup/test-woe-gpt-j-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en woe -mn llama-3-3b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8  > jobs/nohup/test-woe-llama-3-3b-UniEdit-1.log 2>&1 &

python test_editor.py -et llm -en woe -mn gpt2-xl -sen 1 -dvc "cuda:1" -ckpt none -dn UniEdit -dsn 5000 -rs 16  > jobs/nohup/test-woe-gpt2-xl-UniEdit-1.log 2>&1 &
python test_editor.py -et llm -en woe -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16  > jobs/nohup/test-woe-gpt-j-UniEdit-1.log 2>&1 &
python test_editor.py -et llm -en woe -mn llama-3-3b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16  > jobs/nohup/test-woe-llama-3-3b-UniEdit-1.log 2>&1 &

