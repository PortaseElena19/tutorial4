import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generăm un set de date  
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# 1. Instanțiem modelul AgglomerativeClustering
n_clusters = 3  
model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')

# 2. Rulăm modelul pentru a obține etichetele clusterelor
labels = model.fit_predict(X)

# 3. Calculăm matricea de linkage pentru dendrogramă
Z = linkage(X, method='ward')

# 4. Dendrograma
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrograma pentru AgglomerativeClustering')
plt.xlabel('Indexul punctelor')
plt.ylabel('Distanța')
plt.show()

# Vizualizarea clusterelor 
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.title('Clusterele formate prin AgglomerativeClustering')
plt.xlabel('Trăsătura 1')
plt.ylabel('Trăsătura 2')
plt.show()

