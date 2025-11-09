from autograd import grad
import matplotlib.pyplot as plt
import numpy as np

def model(x, w):
    return w[0] + np.dot(x.T, w[1:])

def mse(x, y):
    return sum() / x.shape(0)
