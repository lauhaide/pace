export PYTHONPATH=$PYTHONPATH:codebase/OpenRLHF-RAG

python -u codebase/reward/answer_reflection_reward.py --template_type qwen_chat --data_path codebase/data/training_set/train-orlhf-musique.jsonl --reward_pretrain Qwen/Qwen2.5-1.5B --log_file codebase/results/qwen.jsonl --port 1278