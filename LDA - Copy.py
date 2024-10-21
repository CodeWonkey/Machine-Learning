# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

# Generate a synthetic 2D dataset with more distributed data
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=2, 
                           class_sep=1.5, n_classes=2, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform Linear Discriminant Analysis
lda = LDA()
lda.fit(X_train, y_train)

# Define the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict the label for each point in the mesh
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Scatter plot of original data
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap='coolwarm')

# Add labels and title
plt.title("Linear Discriminant Analysis (LDA) with More Distributed Data")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the projection of the data onto the LDA component
coef = lda.coef_.ravel()  # LDA coefficients
intercept = lda.intercept_  # LDA intercept
x_vals = np.array([x_min, x_max])
y_vals = -(coef[0] * x_vals + intercept) / coef[1]  # Equation of LDA decision boundary

# Plot the LDA line
plt.plot(x_vals, y_vals, color='black', linestyle='--', label='LDA Component')

# Show the plot with the decision boundary and LDA component
plt.legend()
plt.grid(True)
plt.show()