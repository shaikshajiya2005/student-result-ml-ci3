import json
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate sample student result data
X, y = make_classification(
    n_samples=300,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    random_state=42
)

# Save generated dataset
data = pd.DataFrame(X, columns=[
    "Study_Hours",
    "Attendance",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Score"
])

data["Result"] = y
data.to_csv("student_results.csv", index=False)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save trained model
joblib.dump(model, "student_result_model.pkl")

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model training completed.")
print("Model Accuracy:", round(accuracy, 4))
print("Dataset saved as student_results.csv")
