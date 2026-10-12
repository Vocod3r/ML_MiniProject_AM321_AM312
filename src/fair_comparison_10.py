import numpy as np
import pandas as pd
import time
import os

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

data = np.load("data/dataset_10classes.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

# Create a balanced 6,000-sample training subset
rng = np.random.default_rng(42)
selected = []

for label in np.unique(y_train):
    class_indices = np.where(y_train == label)[0]
    selected.extend(rng.choice(class_indices, 600, replace=False))

selected = np.array(selected)
rng.shuffle(selected)

X_small = X_train[selected]
y_small = y_train[selected]

models = {
    "Logistic Regression": LogisticRegression(max_iter=300, solver="lbfgs"),
    "Linear SVM": LinearSVC(C=1.0, max_iter=3000, random_state=42),
    "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Polynomial SVM": SVC(kernel="poly", degree=3, C=1.0, gamma="scale"),
    "Sigmoid SVM": SVC(kernel="sigmoid", C=1.0, gamma="scale")
}

results = []

for name, model in models.items():
    print(f"\nTraining {name} on 6,000 samples...")

    start = time.time()
    model.fit(X_small, y_small)
    elapsed = time.time() - start

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    results.append({
        "Model": name,
        "Validation Accuracy (%)": round(val_acc * 100, 2),
        "Test Accuracy (%)": round(test_acc * 100, 2),
        "Training Time (s)": round(elapsed, 2),
        "Training Samples": len(X_small)
    })

    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Training time: {elapsed:.2f} seconds")

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("results/fair_comparison_10classes.csv", index=False)

print("\n===== 10-CLASS MODEL COMPARISON =====")
print(df.to_string(index=False))
print("\nSaved to results/fair_comparison_10classes.csv")
