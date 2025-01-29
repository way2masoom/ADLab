
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data from Table 1
hours = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
                  2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]).reshape(-1, 1)
pass_ = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# Fit logistic regression model
model = LogisticRegression()
model.fit(hours, pass_)

# Predict probabilities
hours_range = np.linspace(0, 6, 300).reshape(-1, 1)
probabilities = model.predict_proba(hours_range)[:, 1]

# Plot the logistic regression curve
plt.figure(figsize=(10, 6))
plt.scatter(hours, pass_, color='blue', label='Data points')
plt.plot(hours_range, probabilities, color='red', label='Logistic regression curve')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.title('Logistic Regression Curve')
plt.legend()
plt.grid(True)
plt.show()