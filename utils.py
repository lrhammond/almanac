
from torch import clamp, isnan, tensor

def flatten(l):

    return [item for sublist in l for item in sublist]

def denom(x):

    return x if x != 0.0 else 1.0

def clip(x, min_x=-100.0, max_x=100.0):

    return clamp(x, min_x, max_x)

def remove_nans(x):

    if x == None:
        return x
    if not isinstance(x, int):
        x[isnan(x)] = 0
    return x

def plot(data):

    pass

