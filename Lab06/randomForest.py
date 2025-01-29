'''
Experiment:
 Implement a random forest classifier to predict customer churn in a telecom dataset.
Accuracy:  Find overall prediction accuracy.
Confusion Matrix: Breakdown of true/false positives and negatives.
Classification Report: Includes precision, recall, and F1-score for both churned and retained classes.
Note: You can use a real-world dataset such as the Telco Customer Churn Dataset available on Kaggle or create a similar dataset for practice.
'''


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
# Replace 'telecom_churn.csv' with the path to your dataset
data = pd.read_csv('Lab06/telecom_churn.csv')

# Data preprocessing
# Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Handle missing values (if any)
data = data.fillna(method='ffill')  # Replace with appropriate strategy

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Define feature set (X) and target (y)
X = data.drop(columns=['churn'])  # Replace 'churn' with the name of your target column
y = data['churn']

# Normalize numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
# 1. Overall Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")

# 2. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# 3. Classification Report
class_report = classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])
print("\nClassification Report:")
print(class_report)