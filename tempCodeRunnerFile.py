import networkx as nx
import EoN

G = nx.barabasi_albert_graph(500, 5)

tmax = 1
tau = 0.1 # transmission rate
gamma = 1.0 # recovery rate
rho = 0.1 # random fraction of initially infected

#sim = EoN.fast_SIS(G, tau, gamma, rho=rho, return_full_data=True)
#print(sim.get_statuses())

total_centrality = 0
print(nx.degree_centrality(4))