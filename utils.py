''' author: samtenka
    change: 2019-06-12
    create: 2019-06-12
    descrp: simple example of torch autograd
'''

import torch
import functools
import memory_profiler
import time



################################################################################
#           0. TORCH                                                           #
################################################################################

device, _ = (
    (torch.device("cuda:0"), torch.device("cuda:1"))
    if torch.cuda.is_available() else
    (torch.device("cpu"), torch.device("cpu"))
) 



################################################################################
#           1. MATH                                                            #
################################################################################

prod = lambda seq: functools.reduce(lambda a,b:a*b, seq, 1) 



################################################################################
#           2. RESOURCE USAGE                                                  #
################################################################################

start_time = time.time()
secs_endured = lambda: (time.time()-start_time) 
megs_alloced = lambda: memory_profiler.memory_usage(-1, interval=0.001, timeout=0.0011)[0]



################################################################################
#           3. ANSI COMMANDS                                                   #
################################################################################

class Colorizer(object):
    def __init__(self):
        self.ANSI_by_name = {
            '@R ': '\033[31m',
            '@G ': '\033[32m',
            '@Y ': '\033[33m',
            '@B ': '\033[34m',
            '@W ': '\033[37m',
            '@^ ': '\033[1A',
        }
        self.text = ''

    def __add__(self, rhs):
        assert type(rhs) == type(''), 'expected types (Colorizer + string)'
        for name, ansi in self.ANSI_by_name.items():
            rhs = rhs.replace(name, ansi)
        self.text += rhs
        return self

    def __str__(self):
        rtrn = self.text 
        self.text = ''
        return rtrn

CC = Colorizer()

if __name__=='__main__':
    print(CC + 'moo')
    print(CC + '@R moo')
    print(CC + 'hi @W moo' + 'cow')
