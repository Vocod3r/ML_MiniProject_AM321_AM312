import numpy as np
import time
import os

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = np.load("data/dataset_3classes.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]

# Use the same 6,000 samples for a fair comparison
X_small, _, y_small, _ = train_test_split(
    X_train,
    y_train,
    train_size=6000,
    random_state=42,
    stratify=y_train
)

results = []

for C in [0.1, 1, 10]:
    print(f"\nTesting RBF SVM with C={C}")

    model = SVC(kernel="rbf", C=C, gamma="scale")

    start = time.time()
    model.fit(X_small, y_small)
    elapsed = time.time() - start

    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)

    results.append((C, accuracy, elapsed))

    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Training time: {elapsed:.2f} seconds")

best = max(results, key=lambda x: x[1])

os.makedirs("results", exist_ok=True)

with open("results/rbf_tuning.txt", "w") as f:
    f.write("RBF SVM Hyperparameter Tuning\n\n")
    for C, accuracy, elapsed in results:
        f.write(
            f"C={C}, Validation Accuracy={accuracy:.4f}, "
            f"Training Time={elapsed:.2f}s\n"
        )
    f.write(f"\nBest C: {best[0]}\n")
    f.write(f"Best validation accuracy: {best[1]:.4f}\n")

print("\n===== BEST CONFIGURATION =====")
print(f"C: {best[0]}")
print(f"Validation accuracy: {best[1]:.4f}")
print("\nResults saved!")
