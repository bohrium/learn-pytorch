''' author: samtenka
    change: 2019-06-12
    create: 2019-06-12
    descrp: simple example of torch autograd
'''

import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

device = get_device()
