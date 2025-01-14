import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Given data
data = {
    'y': [140, 155, 159, 179, 192, 200, 212, 215],
    'x1': [60, 62, 67, 70, 71, 72, 75, 78],
    'x2': [22, 25, 24, 20, 15, 14, 14, 11]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features (x1, x2) and target (y)
X = df[['x1', 'x2']]
y = df['y']

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Get the coefficients (beta1, beta2) and the intercept (beta0)
beta0 = model.intercept_
beta1, beta2 = model.coef_

# Print the linear regression equation
print(f"The regression equation is: y = {beta0:.2f} + {beta1:.2f}*x1 + {beta2:.2f}*x2")

