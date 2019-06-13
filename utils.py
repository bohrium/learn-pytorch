''' author: samtenka
    change: 2019-06-12
    create: 2019-06-12
    descrp: simple example of torch autograd
'''

import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0"), torch.device("cuda:1")
    else:
        return torch.device("cpu"), torch.device("cpu")

device, _ = get_device()
