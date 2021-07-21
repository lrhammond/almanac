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

### MARKOV GAME 3 ###

# Number of players
num_players = 5

# State space
state_space = [tt(i) for i in range(14)]

# Action spaces
action_spaces = [('a','b'), ('x','y'), ('p','q','r'), ('i', 'j'), ('u','v')]

# Dists
d0 = Categorical(tt([ 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.5, 0.0, 0.0, 0.1, 0.0 ]))
d1 = Categorical(tt([ 0.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.1, 0.0 ]))
d2 = Categorical(tt([ 0.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1 ]))
d3 = Categorical(tt([ 0.1, 0.1, 0.0, 0.2, 0.1, 0.1, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0 ]))
d4 = Categorical(tt([ 0.0, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1 ]))
d5 = Categorical(tt([ 0.5, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0 ]))
d6 = Categorical(tt([ 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.3, 0.0, 0.0 ]))
d7 = Categorical(tt([ 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.4, 0.0, 0.0, 0.3 ]))
d8 = Categorical(tt([ 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.2 ]))

# Initial state dist
initial = d7

# Transition function
def transition(s, a):

    if s == tt(0):
        if a[0] == 'a' and a[1] == 'y':
            return d0.sample()
        elif a == ('b','y','p'):
            return d1.sample()
        elif a == ('b','y','q'):
            return d3.sample()
        else:
            return d6.sample()
    elif s == tt(1):
        if a[2] == 'r':
            return d4.sample()
        elif a == ('a','x','q'):
            return d2.sample()
        else:
            return tt(4)
    elif s == tt(2):
        if a == ('b','y','p'):
            return d5.sample()
        elif a == ('b','x','q'):
            return d1.sample()
        elif a == ('a','y','r'):
            return d2.sample()
        elif a == ('a','x','q'):
            return d6.sample()
        else:
            return d4.sample()
    elif s == tt(3):
        if a[1] == 'p':
            return d0.sample()
        elif a[1] == 'q':
            return d2.sample()
        else:
            return tt(1)
    elif s == tt(4):
        if a == ('b','y','p'):
            return d5.sample()
        elif a == ('b','x','q'):
            return d1.sample()
        elif a == ('a','y','r'):
            return d2.sample()
        else:
            return tt(4)
    elif s == tt(5):
        if a[0] == 'a':
            return d3.sample()
        else:
            return d6.sample()
    elif s == tt(6):
        if a == ('a','x','p'):
            return d4.sample()
        else:
            return tt(6)
    elif s == tt(7):
        if a[0] == 'x':
            return d1.sample()
        else:
            return d0.sample()
    elif s == tt(8):
        if a[1] == 'x' and a[2] == 'r':
            return d5.sample()
        elif a[1] == 'y' and a[2] == 'q':
            return d4.sample()
        else:
            return d3.sample()
    elif s == tt(9):
        if a[1] == 'y':
            return d1.sample()
        elif a == ('b','x','p'):
            return d2.sample()
        else:
            return tt(2)
    # elif s == tt(10):
        # TODO
    # elif s == tt(11):
        # TODO    
    # elif s == tt(12):
        # TODO        
    # elif s == tt(13):
        # TODO
        
# Labelling function # TODO
def labeller(s):

    return None

# Reward functions
def reward_1(s_1, a, s_2):

    pass