import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

DATA_PATH = "data/dataset_50classes.npz"
RESULT_PATH = "results/fair_comparison_50classes.csv"
SEED = 42
SAMPLES_PER_CLASS = 120


def main():
    print("Loading 50-class dataset...")
    data = np.load(DATA_PATH)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    classes = data["classes"]
    rng = np.random.default_rng(SEED)

    # Select the same number of training examples from each class.
    selected_indices = []

    for label in np.unique(y_train):
        indices = np.where(y_train == label)[0]
        selected_indices.extend(
            rng.choice(indices, SAMPLES_PER_CLASS, replace=False)
        )

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    X_small = X_train[selected_indices]
    y_small = y_train[selected_indices]

    print("Training subset:", X_small.shape)
    print("Classes:", len(classes))
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    # Keep validation and test data unchanged.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=300, solver="lbfgs"
        ),
        "Linear SVM": LinearSVC(C=1, max_iter=3000),
        "RBF SVM": SVC(kernel="rbf", C=1, gamma="scale"),
        "Polynomial SVM": SVC(
            kernel="poly", degree=3, C=1, gamma="scale"
        ),
        "Sigmoid SVM": SVC(kernel="sigmoid", C=1, gamma="scale"),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()

        model.fit(X_small, y_small)
        training_time = time.time() - start

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Training time: {training_time:.2f} seconds")

        results.append({
            "Model": name,
            "Training Samples": len(y_small),
            "Classes": len(classes),
            "Validation Accuracy": val_acc,
            "Test Accuracy": test_acc,
            "Training Time (seconds)": round(training_time, 2),
        })

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)
    df = df.sort_values("Test Accuracy", ascending=False)
    df.to_csv(RESULT_PATH, index=False)

    print("\n=== FINAL 50-CLASS COMPARISON ===")
    print(df.to_string(index=False))
    print(f"\nSaved results to {RESULT_PATH}")


if __name__ == "__main__":
    main()