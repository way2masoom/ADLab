# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Data from the table
data = {
    "No_of_competitors": [1, 1, 2, 3, 3, 5, 6],
    "Sale": [3600, 3300, 3100, 2900, 2700, 2300, 1800],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df["No_of_competitors"], df["Sale"], color="blue", label="Data Points")
plt.title("Scatter Plot: Competitors vs Sale Volume")
plt.xlabel("Number of Competitors")
plt.ylabel("Sales Volume")
plt.grid()
plt.legend()
plt.show()

# Linear Regression Model
X = np.array(df["No_of_competitors"]).reshape(-1, 1)  # Reshape for sklearn
y = np.array(df["Sale"])

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Model parameters
print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Predictions
y_pred = model.predict(X)

# Plotting regression line
plt.figure(figsize=(8, 5))
plt.scatter(df["No_of_competitors"], df["Sale"], color="blue", label="Data Points")
plt.plot(df["No_of_competitors"], y_pred, color="red", label="Regression Line")
plt.title("Linear Regression: Competitors vs Sale Volume")
plt.xlabel("Number of Competitors")
plt.ylabel("Sales Volume")
plt.legend()
plt.grid()
plt.show()