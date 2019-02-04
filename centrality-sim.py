import networkx as nx
import EoN
from scipy import stats
import matplotlib.pyplot as plt

# Takes in a graph, as well as the number of simulations to run, and outputs the 
# number of times each node ended up infected
def times_infected(G, N):
	tmax = 1
	tau = 0.9 # transmission rate
	gamma = 0.5 # recovery rate
	rho = 0.1 # random fraction of initially infected

	infected_hist = {}
	for i in range(len(G)):
		infected_hist[i] = 0
	
	for iteration in range(N):
		print("Iteration - " + str(iteration) + ".....")
		sim = EoN.fast_SIS(G, tau, gamma, rho=rho, return_full_data=True)
		results = sim.get_statuses()
		for node in results:
			if results[node] == 'I':
				infected_hist[node] = infected_hist[node] + 1
	return infected_hist
	
def centrality_z(G):
	"""Takes in a graph, and returns an array of zscores corresponding to each 
	node's relative centrality
	"""


G = nx.barabasi_albert_graph(1000, 8)


#total_centrality = 0
#centrality_dict = nx.degree_centrality(G)

#for node in centrality_dict:
#	total_centrality += centrality_dict[node]

#mean_centrality = total_centrality / len(centrality_dict)

#infected_hist = {}
#for i in range(500):
#	infected_hist[i] = 0

#for iteration in range(iterations):

#nfection_hist = times_infected(G, 1000)
#centrality_dict = nx.betweenness_centrality(G)
#infection_hist_array = []
#centrality = []
#for i in range(len(G)):
#	infection_hist_array.append(infection_hist[i])
#	centrality.append(centrality_dict[i])
#infection_zs = stats.zscore(infection_hist_array)


#plot = plt.scatter(x = centrality, y=infection_zs)
#plt.show()

