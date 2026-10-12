import numpy as np
import time
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = np.load("data/dataset_3classes.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]
classes = data["classes"]

print("Dataset loaded!")
print("Training samples:", len(X_train))

# Train Logistic Regression
print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=300, solver="lbfgs")

start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start

# Validation
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)

# Testing
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\n===== RESULTS =====")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test,
    test_predictions,
    target_names=classes,
    zero_division=0
))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, test_predictions))

# Save results
os.makedirs("results", exist_ok=True)

with open("results/logistic_regression.txt", "w") as f:
    f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, test_predictions)))

print("\nResults saved to results/logistic_regression.txt")
