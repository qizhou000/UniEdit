# ft gpt2-xl UniEdit
# python test_editor.py -et llm -en ft -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-ft-gpt2-xl-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt2-xl -sen 10 -dvc "cuda:1" -ckpt none -dn UniEdit -dsn 1000 > jobs/nohup/test-ft-gpt2-xl-UniEdit-10.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt2-xl -sen 50 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 1500 > jobs/nohup/test-ft-gpt2-xl-UniEdit-50.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt2-xl -sen 100 -dvc "cuda:1" -ckpt none -dn UniEdit -dsn 2000 > jobs/nohup/test-ft-gpt2-xl-UniEdit-100.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt2-xl -sen 500 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 2500 > jobs/nohup/test-ft-gpt2-xl-UniEdit-500.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt2-xl -sen 1000 -dvc "cuda:1" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-ft-gpt2-xl-UniEdit-1000.log 2>&1 &
# ft gpt-j UniEdit
# python test_editor.py -et llm -en ft -mn gpt-j -sen 1 -dvc "cuda:1" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-ft-gpt-j-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt-j -sen 10 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 1000 > jobs/nohup/test-ft-gpt-j-UniEdit-10.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt-j -sen 50 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 1500 > jobs/nohup/test-ft-gpt-j-UniEdit-50.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt-j -sen 100 -dvc "cuda:1" -ckpt none -dn UniEdit -dsn 2000 > jobs/nohup/test-ft-gpt-j-UniEdit-100.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt-j -sen 500 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 2500 > jobs/nohup/test-ft-gpt-j-UniEdit-500.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt-j -sen 1000 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-ft-gpt-j-UniEdit-1000.log 2>&1 &
# ft llama-3-8b UniEdit
# python test_editor.py -et llm -en ft -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-ft-llama-3-8b-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ft -mn llama-3-8b -sen 10 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 1000 > jobs/nohup/test-ft-llama-3-8b-UniEdit-10.log 2>&1 &
# python test_editor.py -et llm -en ft -mn llama-3-8b -sen 50 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 1500 > jobs/nohup/test-ft-llama-3-8b-UniEdit-50.log 2>&1 &
# python test_editor.py -et llm -en ft -mn llama-3-8b -sen 100 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 2000 > jobs/nohup/test-ft-llama-3-8b-UniEdit-100.log 2>&1 &
# python test_editor.py -et llm -en ft -mn llama-3-8b -sen 500 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 2500 > jobs/nohup/test-ft-llama-3-8b-UniEdit-500.log 2>&1 &
# python test_editor.py -et llm -en ft -mn llama-3-8b -sen 1000 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 > jobs/nohup/test-ft-llama-3-8b-UniEdit-1000.log 2>&1 &


# other seed
# python test_editor.py -et llm -en ft -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8 # > jobs/nohup/test-ft-gpt2-xl-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt-j -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8 # > jobs/nohup/test-ft-gpt-j-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ft -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 8 # > jobs/nohup/test-ft-llama-3-8b-UniEdit-1.log 2>&1 &

# python test_editor.py -et llm -en ft -mn gpt2-xl -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16 # > jobs/nohup/test-ft-gpt2-xl-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ft -mn gpt-j -sen 1 -dvc "cuda:1" -ckpt none -dn UniEdit -dsn 5000 -rs 16  > jobs/nohup/test-ft-gpt-j-UniEdit-1.log 2>&1 &
# python test_editor.py -et llm -en ft -mn llama-3-8b -sen 1 -dvc "cuda:0" -ckpt none -dn UniEdit -dsn 5000 -rs 16 # > jobs/nohup/test-ft-llama-3-8b-UniEdit-1.log 2>&1 &

