import networkx as nx

import math
import numpy as np

import node2vec
from sklearn.cluster import KMeans
import gym
from simpleSIS import simpleSIS
from gym import spaces
from gym.utils import seeding


class Epi_Env(gym.Env):
    def __init__(self, graph = nx.karate_club_graph(), action_clusters = None, max_timesteps = 20,
                 beta_hi = 0.3, beta_low = 0.1, delta_low = 0.2, delta_hi = 0.4, B = 0.5, F = -200):
        
        if action_clusters == None:
            action_clusters = len(graph)

        self.min_action = -1 
        self.max_action = 1
        self.graph = graph 
        self.action_space = action_clusters
        self.observation_space = len(graph)
        self.sim = simpleSIS(graph, 0.2, 0.2, 0.2)
        self.max_timesteps = max_timesteps
        self.state = np.array(self.sim.get_status())
        
        self.F = F
        self.B = B
        self.beta_hi = beta_hi
        self.beta_low = beta_low

        self.delta_hi = delta_hi
        self.delta_low = delta_low
        self.generate_cluster_map()
    
    def new_sim(self):
        return simpleSIS(self.graph, 0.2, 0.2, 0.2)
    
    def step(self, action):
        step_cost = 0
        reward = 0
        done = False

        for node in range(len(self.graph)):
            node_action = self.sigmoid(action[self.cluster_map[node]])
            if self.sim.status_list[node] == 1:
                new_rec_rate = self.delta_low + (node_action * (self.delta_hi - self.delta_low))
                self.sim.change_rec_rate(node, new_rec_rate)
                step_cost += self.delta_cost(self.sim.rec_rates[node])
            if self.sim.status_list[node] == 0:
                new_inf_rate = self.beta_hi - (node_action * (self.beta_hi - self.beta_low))
                self.sim.change_inf_rate(node, new_inf_rate)
                step_cost += self.beta_cost(self.sim.inf_rates[node])

        #print(self.sim.inf_rates)
        #print(self.sim.rec_rates)
        self.sim.time_step()

        if (self.sim.num_inf > self.sim.N / 2):
            done = True
            reward += self.F
            #print("Died at {}".format(self.sim.t))
            self.last_survived = self.sim.t
            self.reset()
        else:
            if self.sim.t >= self.max_timesteps:
                done = True
                self.last_survived = self.sim.t
                print("Survived")
            self.last_cost = step_cost
            reward = 55 + (((self.sim.N * self.B) - step_cost) * 0.25)
            #print("survied to ", self.sim.t)
        state = np.array(self.sim.status_list)


        return state, reward, done, {}
    
    def reset(self):
        self.sim = self.new_sim()
        return np.array(self.sim.status_list)

    
    def generate_cluster_map(self):
        n2v = node2vec.Node2Vec(self.graph, dimensions=64, walk_length=30, num_walks=200, workers=5)
        model = n2v.fit(window=10, min_count=1, batch_words=4)

        X = []
        for i in range(len(self.graph)):
            X.append(model.wv[str(i)])
        
        kmeans = KMeans(n_clusters=self.action_space, random_state = 0).fit(X)
        self.cluster_map = {}
        for node in range(len(self.graph)):
            self.cluster_map[node] = kmeans.labels_[node]

    def beta_cost(self, beta):
        """
        Cost of an infection change investment
        """
        return ((beta**-1 - self.beta_hi**-1) / (self.beta_low**-1 - self.beta_hi**-1))
    
    def delta_cost(self, delta):
        """
        Cost of an infection change investment
        """
        return (((1 - delta**-1) - (1 - self.delta_low**-1)) / ((1 - self.delta_hi**-1) - (1 - self.delta_low**-1)))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    def render(self):
        print(self.sim.status_list)
    