import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = np.load("data/dataset_3classes.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]
classes = data["classes"]

print("Training Logistic Regression...")
model = LogisticRegression(max_iter=300, solver="lbfgs")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

os.makedirs("results", exist_ok=True)

# Plot confusion matrix
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=200)
plt.close()

# Find misclassified images
wrong_indices = np.where(predictions != y_test)[0]

print(f"\nTotal test images: {len(y_test)}")
print(f"Misclassified images: {len(wrong_indices)}")
print(f"Test accuracy: {np.mean(predictions == y_test):.4f}")

# Display up to 15 incorrect predictions
count = min(15, len(wrong_indices))

fig, axes = plt.subplots(3, 5, figsize=(12, 8))
axes = axes.flatten()

for i in range(count):
    idx = wrong_indices[i]

    axes[i].imshow(
        X_test[idx].reshape(28, 28),
        cmap="gray"
    )

    axes[i].set_title(
        f"Actual: {classes[y_test[idx]]}\n"
        f"Predicted: {classes[predictions[idx]]}"
    )

    axes[i].axis("off")

for i in range(count, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("results/misclassified_doodles.png", dpi=200)
plt.close()

print("\nSaved:")
print("results/confusion_matrix.png")
print("results/misclassified_doodles.png")
