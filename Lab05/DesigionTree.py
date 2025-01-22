# Decision Tree 

# 1. Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
            
# 2. Load the Iris Dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable (species)

# Convert to DataFrame for better understanding (optional)
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['species'] = y
print("First 5 rows of the Iris dataset:\n", iris_df.head())

# 3. Split the Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# 4. Train a Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("Decision tree model trained successfully.")

# 5. Evaluate the Model
y_pred = dt_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Generate a detailed classification report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:\n", report)

# 6. Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()
