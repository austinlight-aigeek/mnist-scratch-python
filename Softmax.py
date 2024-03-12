import numpy as np

def Softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex)