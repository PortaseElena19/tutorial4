import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage

# Generăm un set de date de exemplu 
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# --- K-Means ---
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# --- DBSCAN ---
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# --- Agglomerative Clustering ---
agglo = AgglomerativeClustering(n_clusters=4, linkage='ward')
agglo_labels = agglo.fit_predict(X)

# Calculăm scorul de siluetă pentru fiecare set de etichete
kmeans_silhouette = silhouette_score(X, kmeans_labels)
dbscan_silhouette = silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
agglo_silhouette = silhouette_score(X, agglo_labels)

# Afișăm scorurile de siluetă 
print(f"Scorul de siluetă pentru K-Means este: {kmeans_silhouette:.3f}")
print(f"Scorul de siluetă pentru DBSCAN este : {dbscan_silhouette:.3f}")
print(f"Scorul de siluetă pentru Agglomerative Clustering este : {agglo_silhouette:.3f}")


kmeans_silhouette_samples = silhouette_samples(X, kmeans_labels)
dbscan_silhouette_samples = silhouette_samples(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else np.array([0] * len(X))
agglo_silhouette_samples = silhouette_samples(X, agglo_labels)

# Creăm un grafic pentru fiecare set de etichete
fig, axes = plt.subplots(3, 1, figsize=(10, 18))

# K-Means
axes[0].set_title('Graficul de silueta pentru K-Means', fontsize=16)
axes[0].barh(np.arange(len(X)), kmeans_silhouette_samples, align='center')
axes[0].set_yticks(np.arange(len(X)))
axes[0].set_xlabel('Coeficient de silueta pentru K-Means')
axes[0].set_ylabel('Indexul Punctelor')

# DBSCAN
axes[1].set_title('Graficul de silueta pentru DBSCAN', fontsize=16)
axes[1].barh(np.arange(len(X)), dbscan_silhouette_samples, align='center')
axes[1].set_yticks(np.arange(len(X)))
axes[1].set_xlabel('Coeficient de silueta pentru DBSCAN')
axes[1].set_ylabel('Indexul Punctelor')

# Agglomerative Clustering
axes[2].set_title('Graficul de silueta pentru Agglomerative Clustering', fontsize=16)
axes[2].barh(np.arange(len(X)), agglo_silhouette_samples, align='center')
axes[2].set_yticks(np.arange(len(X)))
axes[2].set_xlabel('Coeficient de silueta pentru Agglomerative Clustering')
axes[2].set_ylabel('Indexul Punctelor')

plt.tight_layout()
plt.show()


