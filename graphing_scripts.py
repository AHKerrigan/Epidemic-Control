import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("saved_runs.csv", header=None)

start = 20
X = []
print(len(data))
for index, row in data.iterrows():
    X.append(start)
    start += 20
print(X)

plt.plot(X, data)
plt.show()