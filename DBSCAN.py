# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# Generate synthetic dataset with blobs for clustering
n_samples = 300
X, y = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.6, random_state=0)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# Identify core points, border points, and noise
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# Unique labels (clusters)
unique_labels = set(labels)
colors = plt.cm.get_cmap('viridis', len(unique_labels))

# Plotting the clusters and noise
plt.figure(figsize=(10, 6))
for k in unique_labels:
    if k == -1:
        # Noise points are labeled as -1
        col = 'k'
        label = 'Noise'
    else:
        col = colors(k)
        label = f'Cluster {k}'
    
    # Mask for each cluster
    class_member_mask = (labels == k)
    
    # Core points
    xy_core = X[class_member_mask & core_samples_mask]
    plt.plot(xy_core[:, 0], xy_core[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10, label=label)

    # Border points
    xy_border = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy_border[:, 0], xy_border[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=5)

# Add labels and title
plt.title('DBSCAN Clustering of Synthetic Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Show legend
plt.legend(loc='best')

# Display the plot
plt.grid(True)
plt.show()
