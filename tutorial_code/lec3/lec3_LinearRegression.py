# import stuff
from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt


# define model
def model(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T

# mse (mean square error)
def mse(w, x, y):
    # loop to compute cost contribution from each input-output pair
    cost = np.sum((model(x, w) - y) ** 2)
    return cost / float(y.size)

# gradient descent function
def gradient_descent(g, step, max_its, w, p):
    # compute gradient
    gradient = grad(g)

    # gradient descent loop
    weight_history = [w]
    cost_history = [g(w)]

    for k in range(max_its):
        # evaluate gradient
        grad_eval = gradient(w)
        grad_eval_norm = grad_eval / np.linalg.norm(grad_eval)

        # take gradient step
        if step == 'd':  # diminishing step
            alpha = 1 / (k + 1)
        else:            # constant step
            alpha = step
        w = w - alpha * grad_eval_norm

        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))

    return weight_history, cost_history

# create data set and test
x = 10 * np.random.rand(1, 100)
y = 0.5 + 2 * x + np.random.randn(1, 100)

plt.figure(1)
plt.plot(x.T, y.T, 'x')

def c(t):
    return mse(t, x, y)

w = np.array([[1.], [1.]])
a, b = gradient_descent(c, 'd', 100, w, 0)

# plot cost history
plt.figure(0)
plt.plot(b)

# plot fitted line
plt.figure(1)
xp = np.linspace(0, 10, 100)
plt.plot(xp, a[100][0] + a[100][1] * xp)
plt.show()

# print learned weights
print("Learned params:", a[100][0], a[100][1])
