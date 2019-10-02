import networkx as nx
import random
import numpy as np

class simpleSIS:
	def __init__(self, graph, inf_rate, rec_rate, initial_infected):
		self.G = graph
		self.N = len(self.G)

		self.t = 0
		self.base_inf_rate = inf_rate
		self.base_rec_rate = rec_rate

		self.status_list = []
		self.inf_rates = []
		self.rec_rates = []
		self.num_inf = 0
		self.num_sus = 0

		# Infect some number of nodes and set their rates 
		for i in range(self.N):
			if random.uniform(0, 1) < initial_infected:
				self.status_list.append(1)
				self.num_inf += 1
			else:
				self.status_list.append(0)
				self.num_sus += 0
			
			self.inf_rates.append(inf_rate)
			self.rec_rates.append(rec_rate)
		

	def time_step(self):
		new_status = []
		for node in range(self.N):

			# If the node is infected, we check to see if we can recover it
			if self.status_list[node] == 1:
				if random.uniform(0, 1) < self.rec_rates[node]:
					new_status.append(0)
					self.num_inf -= 1
					self.num_sus += 1
				else:
					new_status.append(1)
			# Otherwise, we try and infect the node
			else:
				total_prob = 0
				for neighbor in nx.all_neighbors(self.G, node):
					total_prob += self.status_list[neighbor] * self.inf_rates[node]
				if random.uniform(0, 1) < total_prob:
					new_status.append(1)
					self.num_inf += 1
					self.num_sus -= 1
				else:
					new_status.append(0)
		self.t += 1
		self.status_list = new_status

	def change_inf_rate(self, node, new):
		self.inf_rates[node] = new
	def change_rec_rate(self, node, new):
		self.rec_rates[node] = new
	
	def get_status(self):
		return self.status_list

