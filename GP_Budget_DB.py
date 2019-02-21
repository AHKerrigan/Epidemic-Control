import numpy as np
import cvxpy as cp
import networkx as nx
import cvxopt

N = 50

raw_G = nx.erdos_renyi_graph(N, 0.01, directed=True)
G = nx.adj_matrix(raw_G).todense()
I = np.identity(N)
rec_rate = 1/10
max_rec_rate = 1.5*rec_rate
m = np.real(np.max(np.linalg.eigvals(G)))
bar = rec_rate / m
ba = 2.5*bar # Most networks are slightly viral, not too much
bs = 20*ba/100

epsbar = 1
Delta = max(epsbar, max_rec_rate)

Budget = 1.5

beta = cp.Variable(N, pos=True)
delta = cp.Variable(N, pos=True)
B  = np.ones(N) - beta
D = np.zeros(N) + delta
#v = cp.Variable(pos=True)
#lamb = cp.Variable(pos=True)

obj = cp.Minimize(cp.lambda_max(cp.diag(B) * G - cp.diag(D)))

contraints = []
contraints.append(cp.sum(B) + cp.sum(D) <= Budget)
contraints.append(beta <= 1)
contraints.append(beta >= 0)
contraints.append(delta <= 1)

prob = cp.Problem(obj, contraints)
result = prob.solve()

print(result)
print(nx.degree_centrality(raw_G))
print(beta.value)
print(delta.value)