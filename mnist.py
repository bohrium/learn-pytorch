''' author: samtenka
    change: 2019-06-12
    create: 2019-06-11
    descrp: simple example of mnist training
'''


from utils import device, prod, secs_endured, megs_alloced, CC

import numpy as np
import torch
from torch import conv2d, matmul
from torch.autograd import grad as nabla
from torch.nn.functional import relu, log_softmax, nll_loss

from torchvision import datasets, transforms

BTCH = 6
SIDE = 28
NB_C = 10 

class LeClassifier():
    def __init__(self):
        self.subweight_shapes = [
            (16 ,  1     , 5, 5), 
            (16 , 16     , 5, 5),
            (16 , 4*4*16       ), 
            (10 , 16           )
        ]
        self.subweight_offsets = [
            sum(prod(shape) for shape in self.subweight_shapes[:depth])
            for depth in range(len(self.subweight_shapes)+1) 
        ]
        self.subweight_scales = [
            (shape[0] + prod(shape[1:]))**(-0.5)
            for shape in self.subweight_shapes
        ]
        self.weights = torch.cat([
            torch.randn(prod(shape), requires_grad=True)
            for shape in self.subweight_shapes 
        ])
        self.get_subweight = lambda depth: ( 
            self.subweight_scales[depth] * 
            self.weights[self.subweight_offsets[depth]:
                         self.subweight_offsets[depth+1]]
            .view(self.subweight_shapes[depth])
        )

    def loss_at(self, x, y):
        x = relu(1.0 + conv2d(x, self.get_subweight(0), bias=None, stride=2))
        x = relu(1.0 + conv2d(x, self.get_subweight(1), bias=None, stride=2))
        x = x.view(-1, 4*4*16, 1)
        x = relu(1.0 + matmul(self.get_subweight(2), x))
        x = matmul(self.get_subweight(3), x)
        x = x.view(-1, NB_C)
        logits = log_softmax(x, dim=1)
        loss = nll_loss(logits, y)
        return loss

    def train_for_one_epoch(self, train_loader, learning_rate):
        print()
        losses = [] 
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            loss = self.loss_at(data, target)
            grad = nabla(loss, LC.weights, create_graph=True)[0]
            self.weights -= learning_rate * grad.detach()
        
            losses.append(float(loss.item()))

            if (batch_idx+1) % 10: continue
            print(CC+'@^ '+' '*120)
            print(CC+'@^ {:.0f}% completed\t batch perplexity: @B {:.2f} @W \t @R {:.2f} @W megs allocated'.format(
                100.0 * batch_idx / len(train_loader),
                np.exp(sum(losses)/len(losses)),
                megs_alloced()
            ))
            losses = []

    def test(self, test_loader):
        test_losses = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                test_losses.append(self.loss_at(data, target))

        print(CC+'test perplexity: @G {:.3f} @W on {} samples'.format(
            np.exp(sum(test_losses)/len(test_losses)),
            len(test_loader.dataset),
        ))


def get_data_loader(train=True):
    return torch.utils.data.DataLoader(                                                     
        datasets.MNIST('../data', train=train, download=True, transform=transforms.ToTensor()),      
        batch_size=100,
        shuffle=True                                                                 
    )

LC = LeClassifier()
train_loader, test_loader = (get_data_loader(train_flag) for train_flag in [True, False])
for epoch_index, learning_rate in enumerate([1.0, 0.8, 0.6]):
    LC.train_for_one_epoch(train_loader, learning_rate)
    LC.test(test_loader)
    print(CC+'average @Y {:.2f} @W seconds per epoch'.format(secs_endured()/(epoch_index+1)))
    print('')
