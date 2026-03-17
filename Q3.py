import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Load Dataset
df = pd.read_csv('sales.csv')

# 2. Select Numerical Features
numeric_cols = df.select_dtypes(include=[np.number]).columns
data = df[numeric_cols].dropna()

# (Optional) Reduce size if dataset is large
if len(data) > 2000:
    data = data.sample(2000, random_state=42)

# 3. Standardize Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 4. Elbow Method
inertia = []
k_values = range(1, 7)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(k_values, inertia)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# 5. Apply KMeans (k = 3 based on elbow)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=5, max_iter=100)
clusters = kmeans.fit_predict(scaled_data)

data["Cluster"] = clusters

# 6. Cluster Analysis
print("\nCluster Mean Values:")
print(data.groupby("Cluster").mean())

# Silhouette Score
score = silhouette_score(scaled_data, clusters)
print("\nSilhouette Score:", score)

# 7. PCA Visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

plt.figure()
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Clusters (PCA)")
plt.show()
