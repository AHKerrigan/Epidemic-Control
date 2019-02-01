import networkx as nx
import EoN

# Takes in a graph, as well as the number of simulations to run, and outputs the 
# number of times each node ended up infected
def times_infected(G, N):
	tmax = 1
	tau = 0.1 # transmission rate
	gamma = 1.0 # recovery rate
	rho = 0.01 # random fraction of initially infected

	infected_hist = {}
	for i in range(len(G)):
		infected_hist[i] = 0
	
	for iteration in range(N):
		sim = EoN.fast_SIS(G, tau, gamma, rho=rho, return_full_data=True)
		results = 
		print(results)
	



G = nx.barabasi_albert_graph(500, 5)



#iterations = 50

#sim1 = EoN.fast_SIS(G, tau, gamma, rho=rho, return_full_data=True)
#sim2 = EoN.fast_SIS(G, tau, gamma, rho=rho, return_full_data=True)
#sim3 = EoN.fast_SIS(G, tau, gamma, rho=rho, return_full_data=True)

#total_centrality = 0
#centrality_dict = nx.degree_centrality(G)

#for node in centrality_dict:
#	total_centrality += centrality_dict[node]

#mean_centrality = total_centrality / len(centrality_dict)

#infected_hist = {}
#for i in range(500):
#	infected_hist[i] = 0

#for iteration in range(iterations):

times_infected(G, 1)