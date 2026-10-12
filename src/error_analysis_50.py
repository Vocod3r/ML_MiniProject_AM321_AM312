import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

DATA_PATH = "data/dataset_50classes.npz"

CLASS_NAMES = [
    "airplane", "alarm clock", "ambulance", "angel", "ant",
    "apple", "arm", "axe", "banana", "basketball",
    "bear", "bed", "bee", "bicycle", "bird",
    "book", "brain", "bridge", "bus", "butterfly",
    "cake", "calculator", "calendar", "camel", "camera",
    "candle", "car", "carrot", "cat", "chair",
    "clock", "cloud", "cow", "crown", "cup",
    "dog", "dolphin", "donut", "dragon", "guitar",
    "ear", "envelope", "eye", "face", "fan",
    "feather", "fence", "finger", "fish", "flower"
]

RANDOM_STATE = 42
SAMPLES_PER_CLASS = 120

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading 50-class dataset...")

data = np.load(DATA_PATH)

X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# --------------------------------------------------
# CREATE BALANCED TRAINING SUBSET
# 120 samples per class = 6000 samples total
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

selected_indices = []

for label in np.unique(y_train):
    class_indices = np.where(y_train == label)[0]

    selected = rng.choice(
        class_indices,
        size=min(SAMPLES_PER_CLASS, len(class_indices)),
        replace=False
    )

    selected_indices.extend(selected)

selected_indices = np.array(selected_indices)
rng.shuffle(selected_indices)

X_train_small = X_train[selected_indices]
y_train_small = y_train[selected_indices]

print("\nTraining samples:", len(X_train_small))
print("Number of classes:", len(np.unique(y_train_small)))

# --------------------------------------------------
# TRAIN RBF SVM
# --------------------------------------------------

print("\nTraining RBF SVM...")

model = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)

start_time = time.time()

model.fit(X_train_small, y_train_small)

training_time = time.time() - start_time

print(f"Training time: {training_time:.2f} seconds")

# --------------------------------------------------
# PREDICT TEST DATA
# --------------------------------------------------

print("\nPredicting test labels...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {accuracy * 100:.2f}%")

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

labels = np.unique(np.concatenate([y_test, y_pred]))

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

# Save numerical confusion matrix
class_names_for_labels = [
    CLASS_NAMES[int(label)]
    if 0 <= int(label) < len(CLASS_NAMES)
    else str(label)
    for label in labels
]

cm_df = pd.DataFrame(
    cm,
    index=class_names_for_labels,
    columns=class_names_for_labels
)

cm_df.to_csv("results/confusion_matrix_50classes.csv")

# Plot the full confusion matrix
plt.figure(figsize=(22, 18))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names_for_labels,
    yticklabels=class_names_for_labels
)

plt.title("50-Class RBF SVM Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix_50classes.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# FIND MOST CONFUSED CLASS PAIRS
# --------------------------------------------------

confused_pairs = []

for i, actual_label in enumerate(labels):
    for j, predicted_label in enumerate(labels):

        # Ignore correct predictions on the diagonal
        if i != j and cm[i, j] > 0:
            confused_pairs.append({
                "Actual Class": class_names_for_labels[i],
                "Predicted Class": class_names_for_labels[j],
                "Misclassified Samples": int(cm[i, j])
            })

pairs_df = pd.DataFrame(confused_pairs)

if not pairs_df.empty:
    pairs_df = pairs_df.sort_values(
        "Misclassified Samples",
        ascending=False
    )

    pairs_df.to_csv(
        "results/top_confused_pairs_50classes.csv",
        index=False
    )

    print("\nTop 20 confused class pairs:")
    print(pairs_df.head(20).to_string(index=False))

    # Plot the 10 most confused pairs
    top_pairs = pairs_df.head(10).copy()

    top_pairs["Pair"] = (
        top_pairs["Actual Class"]
        + " → "
        + top_pairs["Predicted Class"]
    )

    plt.figure(figsize=(12, 7))

    plt.barh(
        top_pairs["Pair"][::-1],
        top_pairs["Misclassified Samples"][::-1]
    )

    plt.xlabel("Number of Misclassified Samples")
    plt.ylabel("Actual → Predicted")
    plt.title("Top 10 Most Confused Class Pairs")
    plt.tight_layout()

    plt.savefig(
        "results/top_confused_pairs_50classes.png",
        dpi=300
    )

    plt.close()

# --------------------------------------------------
# VISUALIZE MISCLASSIFIED DOODLES
# --------------------------------------------------

misclassified_indices = np.where(y_pred != y_test)[0]

print(
    f"\nMisclassified test samples: "
    f"{len(misclassified_indices)} out of {len(y_test)}"
)

if len(misclassified_indices) > 0:

    number_to_show = min(25, len(misclassified_indices))

    display_indices = rng.choice(
        misclassified_indices,
        size=number_to_show,
        replace=False
    )

    rows = 5
    cols = 5

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(14, 14)
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, display_indices):

        image = X_test[idx].reshape(28, 28)

        actual_label = int(y_test[idx])
        predicted_label = int(y_pred[idx])

        ax.imshow(image, cmap="gray")

        ax.set_title(
            f"Actual: {CLASS_NAMES[actual_label]}\n"
            f"Predicted: {CLASS_NAMES[predicted_label]}",
            fontsize=8
        )

        ax.axis("off")

    plt.suptitle(
        "Misclassified Doodles: 50-Class RBF SVM",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        "results/misclassified_doodles_50classes.png",
        dpi=300
    )

    plt.close()

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=class_names_for_labels,
    zero_division=0
)

with open(
    "results/classification_report_50classes.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("50-Class RBF SVM Error Analysis\n")
    file.write("=" * 45 + "\n\n")
    file.write(f"Test accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Training samples: {len(X_train_small)}\n")
    file.write(f"Training time: {training_time:.2f} seconds\n\n")
    file.write(report)

print("\nClassification report:")
print(report)

print("\nError analysis completed!")

print("\nFiles saved:")
print("1. results/confusion_matrix_50classes.csv")
print("2. results/confusion_matrix_50classes.png")
print("3. results/top_confused_pairs_50classes.csv")
print("4. results/top_confused_pairs_50classes.png")
print("5. results/misclassified_doodles_50classes.png")
print("6. results/classification_report_50classes.txt")