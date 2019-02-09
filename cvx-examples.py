import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(5)

def loss_fn(X, Y, beta):
	return cp.pnorm(np.matmul(X, beta) - Y, p=2)**p

w = cp.Variable()
b = cp.Variable()

obj = 0
x = np.arange(40)
y = 0.3 * x + 5 + np.random.standard_normal(40)

for i in range(40):
	obj += (w * x[i] + b - y[i]) ** 2

print(obj)
