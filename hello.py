''' author: samtenka
    change: 2019-06-12
    create: 2019-06-11
    descrp: simple example of autograd
'''

from utils import device
from time import time

import numpy as np
import torch
from torchvision import datasets, transforms

weights = torch.autograd.Variable(torch.ones(3, dtype=torch.float, device=device), requires_grad=True)

for i in range(10):
    data = 1e-3 * torch.randn(3, device=device)
    loss = (weights - data).pow(2).sum()
    grad = torch.autograd.grad(loss, weights, create_graph=True)[0] 
    grad_frozen = grad.detach()
    hess_grad = torch.autograd.grad(torch.mul(grad, grad_frozen).sum(), weights)[0]

    weights = weights - 0.25 * grad
    
    print()
    print('G\t{} \t typical {:.3f}'.format(' '.join('%.3f'%x for x in grad_frozen.numpy()), 2.0**(1.0-i)))
    print('HG\t{}\t typical {:.3f}'.format(' '.join('%.3f'%x for x in hess_grad.numpy()),   2.0**(2.0-i)))
