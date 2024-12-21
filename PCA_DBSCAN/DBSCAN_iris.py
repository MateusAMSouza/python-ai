from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Carregando a base de dados
data = load_iris()
x = data.data

# Visualização inicial
plt.scatter(x[:, 0], x[:, 1], cmap="plasma")
plt.title("Before DBSCAN")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.show()

# Aplicação do DBSCAN
dbscan = DBSCAN(eps=0.7, min_samples=8).fit(x)
labels = dbscan.labels_

# Visualização após o DBSCAN
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap="plasma")
plt.title("After DBSCAN")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.show()
