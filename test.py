import networkx as nx
import pickle
graph = nx.watts_strogatz_graph(n=500, k=10, p=0.2)
f = open("Standard_Watts500.pkl", "wb")
pickle.dump(graph, f)