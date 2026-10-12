import numpy as np
import time
import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

data = np.load("data/dataset_3classes.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

X_small, _, y_small, _ = train_test_split(
    X_train,
    y_train,
    train_size=6000,
    random_state=42,
    stratify=y_train
)

print("Training Sigmoid SVM on 6,000 samples...")

model = SVC(kernel="sigmoid", C=1.0, gamma="scale")

start = time.time()
model.fit(X_small, y_small)
training_time = time.time() - start

val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\n===== SIGMOID SVM RESULTS =====")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Training Time: {training_time:.2f} seconds")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

os.makedirs("results", exist_ok=True)

with open("results/sigmoid_svm.txt", "w") as f:
    f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write(f"Confusion Matrix:\n{confusion_matrix(y_test, test_pred)}")

print("\nResults saved!")
