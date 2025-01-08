import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample dataset( This is sample dataset, we can use realtime .csv file dataset)

data = {
    "Area": [750, 800, 1200, 1500, 2000],
    "Rooms": [2, 3, 3, 4, 5],
    "Price": [150000, 180000, 250000, 300000, 400000],
}
df = pd.DataFrame(data)

# Features (Area, Rooms) and Target (Price)

X = df[["Area", "Rooms"]]
y = df["Price"]

# Splitting the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Model evaluation

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Visualizing regression line (scatter plot for one feature - Area)

plt.scatter(df["Area"], df["Price"], color="blue", label="Actual Data")
plt.plot(df["Area"], model.predict(df[["Area", "Rooms"]]), color="red", label="Regression Line")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (in USD)")
plt.title("Simple Linear Regression: House Prices")
plt.legend()
plt.show()