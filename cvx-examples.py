import cvxpy as cp
import random
import math
import matplotlib.pyplot as plt
import ndlib.models.epidemics.SISModel as sis
import ndlib.models.ModelConfig as mc
import networkx as nx
import numpy as np
from cvxpy.atoms.affine import diag as cpdiag
from cvxpy.atoms.atom import Atom

np.random.seed(5)

"""
def cost(beta, delta):
	return cp.sum(beta) + cp.sum(delta)

def decay_rate(B, A, D):
	return cp.lambda_max(cp.matmul(B, A) - D)

N = 100
inf_rate = 0.3
rec_rate = 0.05
desired_decay_rate = -5


#G = nx.erdos_renyi_graph(N, 0.1, directed=True)
G = nx.grid_2d_graph(10, 10)
nx.draw_networkx(G)
plt.show()

adj_matrix = nx.adj_matrix(G).todense()


# Creates random weights for the adjecency matrix
for x in range(N):
	for y in range(N):
		if adj_matrix[x, y] == 1:
			adj_matrix[x, y] = random.randint(1, 101)


beta = cp.Variable(N)
delta = cp.Variable(N)
B = np.diag(np.full(N, inf_rate)) - cp.diag(beta)
D = np.diag(np.full(N, rec_rate)) + cp.diag(delta)

budget = N * (inf_rate + (1 - rec_rate))
contraints = [decay_rate(B, adj_matrix, D) <= desired_decay_rate, delta >= 0, beta >=0, beta <= inf_rate - 0.01]
#ontraints = [cost(beta, delta) <= budget, delta >= 0, beta >=0]

obj = cp.Minimize(cost(beta, delta))
prob = cp.Problem(obj, constraints=contraints)
result = prob.solve(gp=True)

print(delta.value)
print(beta.value)
"""
N = 100
p = 0.2
G = nx.adj_matrix(nx.erdos_renyi_graph(N, p, directed=True)).todense()
I = np.identity(N) # N sized identity matrix


de = 1/10
dehigh = 5*de
m = np.real(np.max(np.linalg.eigvals(G))) # Compute the maximum eigenvalue of the graph

bar = de/m
ba = 2 * bar
bs = 30*ba/100

alpha1 = 1/(1/(1-dehigh) - 1/(1-de))
alpha2 = 1/(1/bs-1/ba)

# Geometric program
print(alpha1)
print(alpha2)

B = cp.Variable(shape=(N))
D = cp.Variable(N)
v = cp.Variable(N)
lamb = cp.Variable()
p = cp.Variable()

obj = cp.Minimize(alpha1 * cp.sum(cp.inv_pos(D)) + alpha2 * cp.sum(cp.inv_pos(B)))
constraints = []
constraints.append((cp.diag(B) * G + cp.diag(D)) * v <= v) 
constraints.append(v >= np.zeros(N))
prob = cp.Problem(obj, constraints)
print(prob.solve(gp=True))
