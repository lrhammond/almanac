import torch.tensor as tt
from torch.distributions import Categorical

import torch

if torch.cuda.is_available():
    print("Using GPU!")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("Using CPU!")
    device = torch.device("cpu")

### MARKOV GAME 1 ###

# Number of players
num_players = 2

# State space
state_space = [tt(i) for i in range(5)]

# Action spaces
action_spaces = [('a','b'),('x','y')]

# Dists
d0 = Categorical(tt([ 0.0, 0.3, 0.1, 0.6 ]))
d1 = Categorical(tt([ 0.9, 0.1, 0.0, 0.0 ]))
d2 = Categorical(tt([ 0.3, 0.1, 0.1, 0.5 ]))
d3 = Categorical(tt([ 0.0, 0.0, 1.0, 0.0 ]))
d4 = Categorical(tt([ 0.2, 0.2, 0.4, 0.2 ]))

# Initial state dist
def initial(states):

    return d3.sample()

# Transition function
def transition(s, a):

    if s == tt(0):
        if a[0] == 'a':
            return d0.sample()
        elif a == ('b','x'):
            return d1.sample()
        else:
            return tt(1)
    elif s == tt(1):
        if a == ('a','x'):
            return d0.sample()
        elif a == ('b','x'):
            return d1.sample()
        elif a == ('a','y'):
            return d2.sample()
        else:
            return d4.sample()
    elif s == tt(2):
        if a[1] == 'y':
            return d2.sample()
        else:
            return d3.sample()
    elif s == tt(3):
        if a[0] == 'b':
            return d4.sample()
        else:
            return tt(3)

# Labelling function
def labeller(s):

    if s == tt(0):
        return ('phi','psi')
    elif s == tt(1):
        return ('phi','chi')
    elif s == tt(2):
        return ()
    elif s == tt(3):
        return ('psi',)

# Reward functions
def reward_1(s_1, a, s_2):

    pass