# Build a logistic regression model to classify emails as spam or not spam.
# Calculate and display:
# Accuracy: Percentage of emails correctly classified.
# Confusion Matrix: Breakdown of true positives, true negatives, false positives, and false negatives.
# Precision, Recall, and F1-score for both spam and non-spam classes, using a classification report.



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset: Replace with your email dataset
# Here we create a dummy dataset for demonstration purposes
data = {
    "email": [
        "Win a free iPhone now!",
        "Meeting at 10 am",
        "Congratulations, you have won a lottery!",
        "Can we reschedule our appointment?",
        "Earn $1000 per day working from home",
        "Don't forget the project deadline",
        "Get cheap loans instantly!",
        "Hi! I am Form Bank I need your OTP to confirm your bank account"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0],  # 1: Spam, 0: Not Spam
}

df = pd.DataFrame(data)

# Split data into features and target
X = df["email"]
y = df["label"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"])
print("Classification Report:")
print(class_report)
