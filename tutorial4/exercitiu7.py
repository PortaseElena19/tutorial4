import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 1. Generarea unui set de date 
np.random.seed(42)

# Generăm 300 de puncte pentru 3 clustere
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Adăugăm câțiva outlieri
outliers = np.random.uniform(low=-10, high=10, size=(10, 2)) 
X = np.vstack([X, outliers])

# 2. Standardizarea datelor 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Aplicarea DBSCAN (alegem eps=0.5 și min_samples=5 pentru exemplele standard)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 4. Identificarea outlierilor 
outliers_dbscan = X[dbscan_labels == -1]

plt.figure(figsize=(10, 6))

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis', label='Clustere')
plt.scatter(outliers_dbscan[:, 0], outliers_dbscan[:, 1], color='red', label='Outlieri', s=100, edgecolors='black')

plt.title('Clustere și Outlieri detectați cu DBSCAN')
plt.xlabel('Trăsătura 1')
plt.ylabel('Trăsătura 2')
plt.legend()
plt.show()

# Analizam outlierii pentru a vedea dacă au sens
print(f"Număr de outlieri: {len(outliers_dbscan)}")
print("Coordonatele outlierilor:")
print(outliers_dbscan)
