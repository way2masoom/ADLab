from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Define the data from the table
data = {
    "CGPA": [9, 8, 9, 7, 8, 9, 7, 9, 8, 8],
    "Interactive": [1, 0, 0, 0, 1, 1, 1, 0, 1, 1],  # Yes = 1, No = 0
    "Practical Knowledge": [2, 1, 0, 0, 1, 1, 1, 2, 1, 0],  # Very Good = 2, Good = 1, Average = 0
    "Skills": [2, 1, 0, 2, 1, 1, 0, 2, 2, 2],  # Good = 2, Moderate = 1, Poor = 0
    "Job Offer": [1, 1, 0, 0, 1, 1, 0, 1, 1, 1]  # Yes = 1, No = 0
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Split features and target variable
X = df.drop(columns="Job Offer")
y = df["Job Offer"]

# Train the decision tree classifier
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True)
plt.title("Decision Tree for Job Offer Classification")
plt.show()                                                                                                           