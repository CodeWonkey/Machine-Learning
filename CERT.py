# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import pair_confusion_matrix
from scipy.stats import mode

# Generate synthetic dataset with blobs for clustering
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Apply different clustering algorithms (KMeans, Agglomerative, DBSCAN)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

agglomerative = AgglomerativeClustering(n_clusters=4)
agglomerative_labels = agglomerative.fit_predict(X)

dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Combine the clustering results into an ensemble
labels_matrix = np.vstack((kmeans_labels, agglomerative_labels, dbscan_labels))

# Find the mode of the labels (most common label for each point)
ensemble_labels, _ = mode(labels_matrix, axis=0)

# Plot the ensemble result
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=ensemble_labels.flatten(), cmap='viridis', s=50, edgecolor='k')
plt.title('Ensemble Clustering (CERT-Inspired)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()