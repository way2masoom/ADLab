import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Given data
data = {
    'Y': [-3.7, 3.5, 2.5, 11.5, 5.7],
    'X1': [3, 4, 5, 6, 2],
    'X2': [8, 5, 7, 3, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features (X1, X2) and target (Y)
X = df[['X1', 'X2']]
y = df['Y']

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Get the coefficients (beta1, beta2) and the intercept (beta0)
beta0 = model.intercept_
beta1, beta2 = model.coef_

# Print the linear regression equation
print(f"The regression equation is: Y = {beta0:.2f} + {beta1:.2f}*X1 + {beta2:.2f}*X2")
