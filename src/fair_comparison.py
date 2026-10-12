import numpy as np
import pandas as pd
import time
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the same dataset for all models
data = np.load("data/dataset_3classes.npz")
subset = np.load("data/fair_train_subset.npz")

X_train = subset["X_train"]
y_train = subset["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

models = {
    "Logistic Regression": LogisticRegression(max_iter=300, solver="lbfgs"),
    "Linear SVM": LinearSVC(C=1.0, max_iter=3000, random_state=42),
    "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Polynomial SVM": SVC(kernel="poly", degree=3, C=1.0, gamma="scale"),
    "Sigmoid SVM": SVC(kernel="sigmoid", C=1.0, gamma="scale")
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    results.append({
        "Model": name,
        "Validation Accuracy (%)": round(val_acc * 100, 2),
        "Test Accuracy (%)": round(test_acc * 100, 2),
        "Training Time (s)": round(training_time, 2),
        "Training Samples": len(X_train)
    })

    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)

df.to_csv("results/fair_model_comparison.csv", index=False)

print("\n===== FAIR MODEL COMPARISON =====")
print(df.to_string(index=False))
print("\nSaved to results/fair_model_comparison.csv")
