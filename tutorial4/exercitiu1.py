import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Pasul 1: Încărcarea unui set de date
# Vom crea un set de date folosind make_blobs pentru un demo mai simplu
X, y = make_blobs(n_samples=100, centers=3, n_features=4, random_state=42)

# Crearea unui DataFrame pentru a manipula datele mai ușor
data = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
data['Target'] = y

# Pasul 2: Verificarea valorilor lipsă
print("Verificare valori lipsă:\n", data.isnull().sum())

# Pasul 3: Preprocesarea datelor
# Dacă sunt necesare, vom scalara datele
scaler = StandardScaler()  
scaled_data = scaler.fit_transform(data[['Feature1', 'Feature2', 'Feature3', 'Feature4']])

# Vom adăuga datele scalate într-un DataFrame nou
scaled_df = pd.DataFrame(scaled_data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
scaled_df['Target'] = data['Target']

# Pasul 4: Rezumatul datelor
print("\nRezumatul datelor (dimensiune, statistici de bază):")
print(scaled_df.describe())

# Vizualizarea datelor pentru a verifica prelucrarea
plt.scatter(scaled_df['Feature1'], scaled_df['Feature2'], c=scaled_df['Target'], cmap='viridis')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Vizualizare set de date preprocesat')
plt.show()

# Pasul 5: Aplicarea KMeans pentru clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Alegem 3 clustere (deoarece știm că sunt 3 centre)
scaled_df['Cluster'] = kmeans.fit_predict(scaled_df[['Feature1', 'Feature2', 'Feature3', 'Feature4']])

# Vizualizarea datelor după aplicarea KMeans
plt.scatter(scaled_df['Feature1'], scaled_df['Feature2'], c=scaled_df['Cluster'], cmap='viridis')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Clustering cu KMeans')
plt.colorbar(label='Cluster')
plt.show()

# Distribuția pe clustere
print("\nDistribuția pe clustere:")
print(scaled_df['Cluster'].value_counts())
