import numpy as np
from sklearn.linear_model import LinearRegression

# Given data
X = np.array([3, 5, 7, 9]).reshape(-1, 1)  # Reshape for single feature
Y = np.array([4, 8, 8, 10])

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, Y)

# Get the coefficients (beta1) and the intercept (beta0)
beta0 = model.intercept_
beta1 = model.coef_[0]

# Print the linear regression equation
print(f"The regression equation is: Y = {beta0:.2f} + {beta1:.2f}*X")

# Predict Y values for X = 11 and X = 13
X_new = np.array([11, 13]).reshape(-1, 1)
Y_new = model.predict(X_new)

print(f"Predicted Y when X = 11: {Y_new[0]:.2f}")
print(f"Predicted Y when X = 13: {Y_new[1]:.2f}")
