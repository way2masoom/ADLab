import numpy as np 
 
# Data from the table 
actual = np.array([112, 113, 114, 115, 112, 121, 122, 114]) 
predicted = np.array([113, 112, 116, 117, 110, 118, 121, 115]) 
 
# Number of months 
n = len(actual) 
 
# Calculate MSE 
mse = np.mean((actual - predicted) ** 2) 
 
# Calculate RMSE 
rmse = np.sqrt(mse) 
 
# Calculate Hybrid Error 
hybrid_error = 0.3 * mse + 0.25 * rmse 
 
# Calculate MAPE 
mape = np.mean(np.abs((actual - predicted) / actual)) * 100 
 
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Hybrid Error:", hybrid_error)
print("Mean Absolute Percentage Error:", mape)