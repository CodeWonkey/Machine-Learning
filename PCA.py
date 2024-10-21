# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# Generate a dataset with three classes for PCA
n_samples = 150
n_features = 5  # Number of features in the dataset
n_informative = 3  # Number of informative features

# Create a synthetic dataset with 3 distinct classes
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, 
                           n_classes=3, n_clusters_per_class=1, random_state=42)

# Apply PCA to reduce data to 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a figure
plt.figure(figsize=(10, 6))

# Scatter plot of PCA transformed data for the three classes
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)

# Add labels and title
plt.title('PCA Bifurcation with Three Different Sets of Points')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Add a legend
legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend1)

# Show the plot
plt.grid(True)
plt.show()