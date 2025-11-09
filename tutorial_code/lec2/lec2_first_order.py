# import auto diff
from autograd import grad                 # 【M-0】需要 ∇g(w)（梯度）；用自动求导得到
import numpy as np
import matplotlib.pyplot as plt

# define function g(w)
def g(w):                                  # 【M-1】定义目标函数 g(w)
    return w**2                            #      数学：g(w) = w^2（凸；最优解 w* = 0）

# gradient descent function
def gradient_descent(g, alpha, max_its, w, p):
    # compute gradient
    gradient = grad(g)                     # 【M-2】构造梯度函数 w ↦ ∇g(w)；此处等价于 ∇g(w)=2w

    # gradient descent loop
    weight_history = [w]                   # 【M-9】记录 {w^k}
    cost_history   = [g(w)]                # 【M-9】记录 {g(w^k)}

    for k in range(max_its):               # 【M-3】for k = 0..max_its-1
        # eval gradient
        grad_eval = gradient(w)            # 【M-4】计算当前梯度：grad_eval = ∇g(w^k)

        # take grad descent step
        w = w - alpha * grad_eval / np.linalg.norm(grad_eval)
                                           # 【M-5】更新：
                                           #  w^{k+1} = w^k - α * ∇g(w^k)/||∇g(w^k)||_2
                                           #  （方向 = 最速下降 -∇g；除范数=单位方向；步长=α）

        if p:                              # 【M-10】可视化：把 (w^{k+1}, g(w^{k+1})) 画在图上
            plt.plot(w, g(w), "kx")

        # record weight and cost
        weight_history.append(w)           # 【M-9】追加 w^{k+1}
        cost_history.append(g(w))          # 【M-9】追加 g(w^{k+1})

    return weight_history, cost_history    # 【M-11】返回迭代轨迹（便于检查收敛）

# generate random initialization
scale = 5                                  # 【M-6】设定初始范围（仅工程细节）
N = 1                                      # 【M-6】问题维度（这里是一维）
w = scale * np.random.rand(N, 1)           # 【M-6】随机初始化 w^0（选择起点）

# plot function and initial (w, g(w))
x = np.linspace(-5, 5, 100)                # 【M-10】画图采样点
plt.plot(x, g(x))                          # 【M-10】画目标函数曲线 y = g(w)
plt.plot(w, g(w), 'kx')                    # 【M-10】标出初始点 (w^0, g(w^0))

# call gradient descent
u = gradient_descent(g, .1, 20, w, 1)      # 【M-7】【M-8】运行 GD：α=0.1；最多 20 次；开启可视化

plt.show()                                 # 【M-10】显示图像

