import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a sample dataset with 1 informative feature for logistic regression
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
logistic_regressor = LogisticRegression()
logistic_regressor.fit(X_train, y_train)

# Generate a range of values for plotting the S-curve
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = logistic_regressor.predict_proba(X_range)[:, 1]  # Probabilities for class 1

# Plotting the data points and the logistic regression S-curve
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_range, y_prob, color='red', label='Logistic Regression S-curve')
plt.xlabel('Feature')
plt.ylabel('Probability')
plt.title('Logistic Regression S-curve')
plt.legend()
plt.show()
