import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics.SISModel as sis
import matplotlib.pyplot as plt

def full_centrality_experiment(g, sim_len, num_experimnts):
	# model selection
	model = sis.SISModel(g)

	N = len(g)

	# Model Configuration
	cfg = mc.Configuration()
	cfg.add_model_parameter('beta', 0.01)
	cfg.add_model_parameter('lambda', 0.005)
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

	between_centrality = []
	eigen_centrality = []

	between_dict = nx.betweenness_centrality(g)
	eigen_dict = nx.eigenvector_centrality(g)

	for x in range(N):
		between_centrality.append(between_dict[x])
		eigen_centrality.append(eigen_dict[x])

	plt.scatter(x = between_centrality, y = infection_hist)
	plt.show()

# Size of network
N = 1000
sim_len = 20
num_experiments = 2000

# Network topology
#g = nx.barabasi_albert_graph(N, 5)
