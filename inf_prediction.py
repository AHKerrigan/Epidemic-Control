import simpleSIS
import networkx as nx
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

np.random.seed(3243)
G = nx.karate_club_graph()
sim = simpleSIS(G, 0.1, 0.3, 0.15)

model = Sequential()
model.add(Dense(100, input_dim=sim.N, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(sim.N, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X = []
y = []
for episode in range(10000):
	print("Episode " + str(episode))
	sim = simpleSIS(G, 0.1, 0.05, 0.15)
	start = sim.get_status()

	for i in range(5):
		sim.time_step()
	
	end = sim.get_status()

	X.append(start)
	y.append(end)

X = np.matrix(X)
y = np.matrix(y)
model.fit(X, y, batch_size = 10000, epochs=10000)
sim = simpleSIS(G, 0.1, 0.3, 0.15)
start = sim.get_status()

for i in range(20):
	sim.time_step()

end = sim.get_status()
print("Real")
print(start)
print("predicted")
print(model.predict(np.matrix(start)))
