# AIML Practical Assignment No. 11
# Title: K-Means Clustering on Wine Dataset

# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 2: Load Wine dataset
wine = load_wine()

X = wine.data
y = wine.target

print("Wine Dataset Loaded Successfully")
print("Dataset Shape:", X.shape)

# Step 3: Convert dataset into DataFrame
df = pd.DataFrame(X, columns=wine.feature_names)

print("\nFirst Five Rows of Dataset:")
print(df.head())

# Step 4: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData Standardized Successfully")

# Step 5: Apply K-Means clustering
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

print("\nK-Means Clustering Completed")

# Step 6: Calculate Silhouette Score
score = silhouette_score(X_scaled, clusters)

print("\nSilhouette Score:", score)

# Step 7: Apply PCA for dimensionality reduction
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Applied Successfully")
print("Reduced Dataset Shape:", X_pca.shape)

print("Explained Variance Ratio:",
      pca.explained_variance_ratio_)

print("Total Explained Variance:",
      sum(pca.explained_variance_ratio_) * 100, "%")

# Step 8: Visualize clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap='viridis',
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering on Wine Dataset using PCA")

plt.colorbar(label="Cluster")
plt.show()

# Step 9: Display final results
print("\nFinal Results")
print("Number of Clusters:", 3)
print("Silhouette Score:", score)

print("\nCluster Analysis:")

for i in range(3):
    print(f"Cluster {i}: {np.sum(clusters == i)} data points")

print("\nExperiment Completed Successfully!")