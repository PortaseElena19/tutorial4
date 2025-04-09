import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs

# Generăm un set de date de exemplu 
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# 1. Aplicăm KMeans (pentru comparație)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
kmeans_labels = kmeans.labels_

# 2. Aplicăm DBSCAN
# Alegem parametrii eps și min_samples
eps = 0.5  # distanța maximă între două puncte pentru a fi considerate în același cluster
min_samples = 5  # numărul minim de puncte într-un cluster

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X)

# 3. Identificăm outlierii (etichetele == -1)
outliers_dbscan = np.sum(dbscan_labels == -1)

# 4. Comparăm numărul de clustere detectate de DBSCAN și outlieri cu soluția K-Means
num_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

# Rezultate pentru DBSCAN
print(f"Numărul de clustere detectate de DBSCAN: {num_clusters_dbscan}")
print(f"Numărul de outlieri detectați de DBSCAN: {outliers_dbscan}")

# Comparăm cu rezultatele K-Means
num_clusters_kmeans = len(set(kmeans_labels))

# Vizualizarea clusterelor KMeans și DBSCAN
plt.figure(figsize=(14, 6))

# KMeans
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.6)
plt.title('Clustere K-Means')
plt.xlabel('Trăsătura 1')
plt.ylabel('Trăsătura 2')

# DBSCAN
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50, alpha=0.6)
plt.title('Clustere DBSCAN')
plt.xlabel('Trăsătura 1')
plt.ylabel('Trăsătura 2')

plt.show()

# Afișăm numărul de clustere și outlieri pentru comparație
print(f"Numărul de clustere detectate de K-Means: {num_clusters_kmeans}")
