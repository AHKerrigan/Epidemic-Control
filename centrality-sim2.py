import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics.SISModel as sis
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def full_centrality_experiment(g, sim_len, num_experimnts, infection_rate, recovery_rate):
	# model selection
	model = sis.SISModel(g)

	N = len(g)

	# Model Configuration
	cfg = mc.Configuration()
	cfg.add_model_parameter('beta', infection_rate)
	cfg.add_model_parameter('lambda', recovery_rate)
	cfg.add_model_parameter("percentage_infected", 0.05)
	model.set_initial_status(cfg)

	# Total infection histogram
	infection_hist = [0] * N

	for i in range(num_experiments):
		print("Experiment # " + str(i))
		model.reset()
		model.iteration_bunch(sim_len)
		for x in range(N):
			infection_hist[x] = infection_hist[x] + model.status[x]
	
	infection_hist = np.log(infection_hist)

	degree_centrality = []
	closeness_centrality = []
	between_centrality = []
	eigen_centrality = []

	degree_dict = nx.degree_centrality(g)
	closeness_dict = nx.closeness_centrality(g)
	between_dict = nx.betweenness_centrality(g)
	eigen_dict = nx.eigenvector_centrality(g)

	for x in range(N):
		degree_centrality.append(degree_dict[x])
		closeness_centrality.append(closeness_dict[x])
		between_centrality.append(between_dict[x])
		eigen_centrality.append(eigen_dict[x])
	
	degree_centrality = stats.zscore(degree_centrality)
	closeness_centrality = stats.zscore(closeness_centrality)
	between_centrality = stats.zscore(between_centrality)
	eigen_centrality = stats.zscore(eigen_centrality)

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.scatter(x = degree_centrality, y=infection_hist, label='Degree', s=10)
	ax1.scatter(x = closeness_centrality, y=infection_hist, label='Closeness', s=10)
	ax1.scatter(x = between_centrality, y = infection_hist, label='Betweeness', s=10)
	ax1.scatter(x = eigen_centrality, y = infection_hist, label='Eigenvector',s=10)
	ax1.legend()
	ax1.set_xlabel('Z-Score of Centrality')
	ax1.set_ylabel('Total Number of Times Infected (log)')
	plt.xlim((-4, 4))
	plt.show()

# Size of network
N = 1000
sim_len = 100
num_experiments = 4000

# Network topology
#g = nx.barabasi_albert_graph(N, 5)
g = nx.erdos_renyi_graph(N, 0.01)
full_centrality_experiment(g, sim_len, num_experiments, 0.01, 0.01)     