import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Define the dataset based on the uploaded image
data = pd.DataFrame({
    'Y': [-3.7, 3.5, 2.5, 11.5, 5.7],
    'X1': [3, 4, 5, 6, 2],
    'X2': [8, 5, 7, 3, 1]
})

# Step 2: Split into features (X1, X2) and target (Y)
X = data[['X1', 'X2']]
y = data['Y']

# Step 3: Train the multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Make predictions
y_pred = model.predict(X)

# Step 5: Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Output results
mse, r2, model.coef_, model.intercept_


# Step 6: Scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', alpha=0.7, label='Predicted vs Actual')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal Fit')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.legend()
plt.grid()
plt.show()