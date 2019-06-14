''' author: samtenka
    change: 2019-06-12
    create: 2019-06-11
    descrp: simple example of mnist training
'''

from utils import device
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(4*4*32, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
                                    # batch x  1 x 28 x 28
        x = F.relu(self.conv1(x))   # batch x 16 x 12 x 12
        x = F.relu(self.conv2(x))   # batch x 32 x  4 x  4
        x = x.view(-1, 4*4*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    print()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20: continue
        print('\033[1A'+' '*120)
        print('\033[1Atrain epoch {}, {:.0f}% completed\t batch perplexity: {:.3f}'.format(
            epoch,
            100.0 * batch_idx / len(train_loader),
            np.exp(float(loss.item()))
        ))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('test perplexity: {:.3f}\t test accuracy: {}/{} ({:.0f}%)'.format(
        np.exp(test_loss),
        correct,
        len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset))
    )

def main():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=50, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=50, shuffle=True
    )

    model = SimpleNet().to(device)

    start = time()

    nb_epochs = 5
    print()
    for epoch in range(nb_epochs):
        lr = 10**(-1.5 - epoch/float(nb_epochs))
        print('learning rate {:.4f}'.format(lr))
        optimizer = optim.SGD(model.parameters(), lr=lr)
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        end = time()
        print('average of {:.1f} seconds per epoch'.format((end-start)/(epoch+1)))
        print()

    end = time()
    print('training completed!')
    print('{:.1f} seconds total'.format(end-start))

    torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
