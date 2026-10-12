import os
import pandas as pd
import matplotlib.pyplot as plt

# Create results folder if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load results for 3, 10, and 50 classes
df3 = pd.read_csv("results/fair_model_comparison.csv")
df10 = pd.read_csv("results/fair_comparison_10classes.csv")
df50 = pd.read_csv("results/fair_comparison_50classes.csv")

# Standardize column names in the 50-class results
df50 = df50.rename(columns={
    "Validation Accuracy": "Validation Accuracy (%)",
    "Test Accuracy": "Test Accuracy (%)",
    "Training Time (seconds)": "Training Time (s)"
})

# Convert 50-class accuracy values from decimals to percentages
df50["Validation Accuracy (%)"] *= 100
df50["Test Accuracy (%)"] *= 100

# Add the number of classes to each dataset
frames = []

for df, n_classes in [
    (df3, 3),
    (df10, 10),
    (df50, 50)
]:
    df = df.copy()
    df["Classes"] = n_classes
    frames.append(df)

# Combine all results
combined = pd.concat(frames, ignore_index=True)

# Save combined results
combined.to_csv(
    "results/combined_model_results.csv",
    index=False
)

# Models included in the comparison
models = [
    "Logistic Regression",
    "Linear SVM",
    "RBF SVM",
    "Polynomial SVM",
    "Sigmoid SVM"
]

# --------------------------------------------------
# GRAPH 1: Test Accuracy Across 3, 10, and 50 Classes
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for model in models:
    subset = combined[
        combined["Model"] == model
    ].sort_values("Classes")

    plt.plot(
        subset["Classes"],
        subset["Test Accuracy (%)"],
        marker="o",
        label=model
    )

plt.xticks([3, 10, 50])
plt.xlabel("Number of Classes")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Across 3, 10 and 50 Classes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/test_accuracy_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# GRAPH 2: Training Time Across 3, 10, and 50 Classes
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for model in models:
    subset = combined[
        combined["Model"] == model
    ].sort_values("Classes")

    plt.plot(
        subset["Classes"],
        subset["Training Time (s)"],
        marker="o",
        label=model
    )

plt.xticks([3, 10, 50])
plt.xlabel("Number of Classes")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time Across 3, 10 and 50 Classes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/training_time_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# GRAPH 3: Test Accuracy Difference Between 3 and 50
# --------------------------------------------------

accuracy3 = (
    combined[combined["Classes"] == 3]
    .set_index("Model")
)

accuracy50 = (
    combined[combined["Classes"] == 50]
    .set_index("Model")
)

accuracy_difference = (
    accuracy3["Test Accuracy (%)"]
    - accuracy50["Test Accuracy (%)"]
).reindex(models)

plt.figure(figsize=(10, 6))

accuracy_difference.plot(kind="bar")

plt.xlabel("Model")
plt.ylabel("Accuracy Difference (percentage points)")
plt.title("Test Accuracy Difference: 3 Classes vs 50 Classes")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/accuracy_difference.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nGraphs created successfully!")

print("\nCombined Model Results:")
print(
    combined[
        [
            "Model",
            "Classes",
            "Test Accuracy (%)",
            "Training Time (s)"
        ]
    ].to_string(index=False)
)

print("\nSaved files:")
print("1. results/combined_model_results.csv")
print("2. results/test_accuracy_comparison.png")
print("3. results/training_time_comparison.png")
print("4. results/accuracy_difference.png")