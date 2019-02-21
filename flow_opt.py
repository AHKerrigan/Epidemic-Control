import networkx as nx
import numpy as np
import cvxpy as cvx

class Edge(object):
	""" A edge in the network with a capacity"""
	def __init__(self, capacity):
		self.capacity = capacity
		self.flow = cvx.Variable()
	
	# Connects two nodes via the edge
	def connect(self, in_node, out_node):
		in_node.edge_flows.append(0-self.flow)
		out_node.edge_flows.append(self.flow)

	# Returns the edge's internal contraints. The flow of the edge cannot exceed capacity
	def constraints(self):
		return self.flow <= self.capacity

class Node(object):
	""" A node in the network"""
	def __init__(self):
		self.accumulation = 0
		self.edge_flows = []

	def calculate_accumulation(self):
		self.accumulation = cvx.sum(self.edge_flows)

	# Returns the node's internal contraints
	def constraints(self):
		return self.accumulation <= 0

# Simple example network
# TO DO: Write functions for reading graph from file or from networkx obeject
# Also write it for random matrix
N = 6

# Source node must be spesified, because it can have as much flow as it can pump
source_node = 0
graph = [[0, 16, 13, 0, 0, 0],
		 [0, 0, 10, 12, 0, 0],
		 [0, 4, 0, 0, 14, 0],
		 [0, 0, 9, 0, 0, 20],
		 [0, 0, 0, 7, 0, 4],
		 [0, 0, 0, 0, 0, 0]]

nodes = []
edges = []

# Create all the nodes in the network first
for node in range(N):
	nodes.append(Node())

# Define all the edges in the network
for i in range(N):
	for j in range(N):
		# If  an edge exists in the graph, create it and append its nodes
		if graph[i][j] != 0:
			new_edge = Edge(graph[i][j])
			new_edge.connect(nodes[j], nodes[i])
			edges.append(new_edge)

# Once all the edges are defined, calculate the accumlation of the network
for node in nodes:
	node.calculate_accumulation()

# Create an array of all the flows, as this is what we will be trying to maximize
flows = []
for edge in edges:
	flows.append(edge.flow)
flow = np.array(flows)

constraints = []
# We create a seperate function for the node contraints since we want to avoid
# accidentally contraining the source node
for i in range(len(nodes)):
	if i != source_node:
		constraints.append(nodes[i].constraints())
for edge in edges:
	constraints.append(edge.constraints())

obj = cvx.Maximize(cvx.sum(flows))
prob = cvx.Problem(obj, constraints=constraints)
print("The max sum of all a_i is", prob.solve())