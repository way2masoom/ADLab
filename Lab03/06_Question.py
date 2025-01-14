import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Given data
data = {
    'number_of_hrs': [2, 3, 4],
    'actual_score': [74, 80, 76],
    'predicted_score': [72, 83, 79]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate R² (using sklearn's r2_score)
r2 = r2_score(df['actual_score'], df['predicted_score'])
print("R² Value:", r2)

# Plotting the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(df['actual_score'], df['predicted_score'], color='blue', alpha=0.7)
plt.plot([min(df['actual_score']), max(df['actual_score'])], 
         [min(df['actual_score']), max(df['actual_score'])], color='red', linestyle='--', label='Ideal Fit')
plt.title('Actual vs Predicted Scores')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.legend()
plt.grid()
plt.show()

# Heatmap of the dataset
correlation_matrix = df[['number_of_hrs', 'actual_score', 'predicted_score']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlations')
plt.show()
