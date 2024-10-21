# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate a sample dataset for regression
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)  # Generate a range of values for smoother line
y_pred = rf_regressor.predict(X_range)

# Plot the data points and the Random Forest regression line
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_range, y_pred, color='red', label='Random Forest Regression Line', linewidth=2)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Random Forest Regression')
plt.legend()
plt.show()