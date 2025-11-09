# import auto diff
from autograd import grad
from autograd import hessian
import numpy as np
import matplotlib.pyplot as plt

def g(w):
    return 3 * w ** 2 + 5 * (w + 1) ** 4

# newton method function
def newtons_method(g, max_its, w, p):
    # compute gradient and hess
    gradient = grad(g)
    hess = hessian(g)
    # set eps for num stability
    epsilon = 10**(-7)
    # newton method loop
    weight_history = [w] # weight history container
    cost_history = [g(w)] # cost history container
    for k in range(max_its):
        # eval gradient and hessian
        grad_eval = gradient(w)
        hess_eval = hess(w)
        # reshape hessian to square mat
        hess_eval.shape = (int((np.size(hess_eval))**(.5)),
                           int((np.size(hess_eval))**(.5)))
        # solve second order system for weight update
        A = hess_eval + epsilon*np.eye(w.size)
        b = grad_eval
        w = np.linalg.solve(A, np.dot(A, w) - b)
        if p:
            plt.plot(w, g(w), "kx")
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history, cost_history

N = 1
scale = 5
w = scale * np.random.rand(N, 1)
x = np.linspace(-5, 5, 1000)
plt.plot(x, g(x))
newtons_method(g, 10, w, 1)

plt.show()