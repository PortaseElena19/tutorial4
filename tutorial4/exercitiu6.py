import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 1. Generarea unui set de date random
np.random.seed(42)

# Simulăm datele pentru 300 de clienți
n_samples = 300

# Frecvența de cumpărare (zile între achiziții, mai mic = mai frecvent)
purchase_frequency = np.random.normal(30, 15, n_samples)

# Cheltuiala medie per achiziție
avg_spending = np.random.normal(50, 20, n_samples)

# Scor de loialitate (scor între 0 și 100)
loyalty_score = np.random.normal(70, 15, n_samples)

# Creăm un DataFrame cu aceste date
data = pd.DataFrame({
    'purchase_frequency': purchase_frequency,
    'avg_spending': avg_spending,
    'loyalty_score': loyalty_score
})

# 2. Standardizarea datelor
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 3. Aplicarea K-Means pentru a crea 4 clustere
kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(data_scaled)

# 4. Analiza clusterelor
cluster_0 = data[data['cluster'] == 0]
cluster_1 = data[data['cluster'] == 1]
cluster_2 = data[data['cluster'] == 2]
cluster_3 = data[data['cluster'] == 3]

# 5. Descrierea clusterelor
print("Cluster 0: Clienți care cheltuiesc mult și au o frecvență de cumpărare mare")
print(cluster_0[['purchase_frequency', 'avg_spending', 'loyalty_score']].mean())
print()

print("Cluster 1: Vânători de oferte, frecvență medie de cumpărare, cheltuiesc mai puțin")
print(cluster_1[['purchase_frequency', 'avg_spending', 'loyalty_score']].mean())
print()

print("Cluster 2: Clienți fideli, frecvență scăzută de cumpărare, cheltuieli mai mari")
print(cluster_2[['purchase_frequency', 'avg_spending', 'loyalty_score']].mean())
print()

print("Cluster 3: Clienți ocazionali, frecvență scăzută de cumpărare și cheltuieli scăzute")
print(cluster_3[['purchase_frequency', 'avg_spending', 'loyalty_score']].mean())
print()

# 6. Vizualizarea clusterelor (scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(cluster_0['purchase_frequency'], cluster_0['avg_spending'], label='Cluster 0', c='blue')
plt.scatter(cluster_1['purchase_frequency'], cluster_1['avg_spending'], label='Cluster 1', c='green')
plt.scatter(cluster_2['purchase_frequency'], cluster_2['avg_spending'], label='Cluster 2', c='red')
plt.scatter(cluster_3['purchase_frequency'], cluster_3['avg_spending'], label='Cluster 3', c='purple')

plt.xlabel('Frecvența de cumpărare')
plt.ylabel('Cheltuiala medie')
plt.title('Segmentele de clienți pe baza frecvenței de cumpărare și cheltuielii medii')
plt.legend()
plt.show()
