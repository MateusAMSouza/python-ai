#!pip install mglearn

import matplotlib.pyplot as plt
import numpy as np
import mglearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carregando a base de dados
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=1)

# Visualização antes do PCA
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
class_0 = wine.data[wine.target == 0]
class_1 = wine.data[wine.target == 1]
class_2 = wine.data[wine.target == 2]
ax = axes.ravel()

for i in range(4):
    _, bins = np.histogram(wine.data[:, i], bins=50)
    ax[i].hist(class_0[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(class_1[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].hist(class_2[:, i], bins=bins, color=mglearn.cm3(1), alpha=.5)
    ax[i].set_title(wine.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["class_0", "class_1", "class_2"], loc="best")
fig.tight_layout()
plt.show()

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(wine.data)

# Aplicação do PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualizando os resultados do PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=wine.target, cmap=plt.cm.Set1, edgecolor='k')
plt.title("PCA do dataset Wine")
plt.xlabel("Primeiro Componente Principal")
plt.ylabel("Segundo Componente Principal")
plt.show()
