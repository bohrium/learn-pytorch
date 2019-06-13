mnist:
	srun --nodes=1 --gres=gpu:8 --partition=learnfair --time=60 python mnist.py
