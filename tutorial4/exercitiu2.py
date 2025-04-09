import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generăm un set de date de exemplu (poți să-l înlocuiești cu datele tale preprocesate)
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# 1. Instanțiem KMeans din sklearn
n_clusters = 4  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# 2. Rulăm algoritmul KMeans pe datele generate (înlocuiește X cu datele tale preprocesate)
kmeans.fit(X)

# Obținem etichetele clusterelor
labels = kmeans.labels_

# 3. Imprimăm un histogram al frecvenței clusterelor
plt.hist(labels, bins=np.arange(n_clusters+1)-0.5, edgecolor='black')
plt.title('Frecvența clusterelor')
plt.xlabel('Cluster')
plt.ylabel('Frecvență')
plt.show()

# 4. Reprezentăm datele într-un scatterplot colorat după etichetele clusterelor
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.title('Clustere K-Means')
plt.xlabel('Trăsătura 1')
plt.ylabel('Trăsătura 2')
plt.show()

