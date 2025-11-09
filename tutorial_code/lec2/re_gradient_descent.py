
import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
def g(w): # define the function
    return w ** 2

def gradient_descent(w, alpha, g, max_k=20):
    w_history = [w]
    g_history = [g(w)]
    gradient = grad(g)
    grad_history = []

    for k in range(max_k):

        grad_eval = gradient(w)
        w = w - alpha * grad_eval / np.linalg.norm(grad_eval)
        w_history.append(w)
        g_history.append(g(w))
        grad_history.append(grad_eval)
        plt.plot(w, g(w), 'kx')

    return w_history, g_history, grad_history
    

scale = 5
N = 1
w = scale * np.random.rand(N, 1)
alpha = 0.1
max_k=20
w_history, g_history, _ = gradient_descent(w, alpha, g, max_k=20)

# plot the image
# 1. draw the function
x = np.linspace(-5 , 5, 100)
plt.plot(x, g(x))
# plt.plot(w_history, g_history)
plt.show()