import os
import pandas as pd

results = [
    {
        "Model": "Logistic Regression",
        "Test Accuracy (%)": 93.17,
        "Training Time (s)": 16.32,
        "Training Samples": 48000
    },
    {
        "Model": "Linear SVM",
        "Test Accuracy (%)": 93.25,
        "Training Time (s)": 30.59,
        "Training Samples": 48000
    },
    {
        "Model": "RBF SVM",
        "Test Accuracy (%)": 95.18,
        "Training Time (s)": 5.98,
        "Training Samples": 6000
    },
    {
        "Model": "Polynomial SVM",
        "Test Accuracy (%)": 94.23,
        "Training Time (s)": 7.05,
        "Training Samples": 6000
    },
    {
        "Model": "Sigmoid SVM",
        "Test Accuracy (%)": 88.47,
        "Training Time (s)": 4.25,
        "Training Samples": 6000
    }
]

df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

df.to_csv("results/model_comparison.csv", index=False)
df.to_string("results/model_comparison.txt", index=False)

print("\n===== MODEL COMPARISON =====")
print(df.to_string(index=False))
print("\nSaved to results/model_comparison.csv")
print("Saved to results/model_comparison.txt")
