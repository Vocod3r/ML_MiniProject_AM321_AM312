import numpy as np
from sklearn.model_selection import train_test_split

data = np.load("data/dataset_3classes.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_small, _, y_small, _ = train_test_split(
    X_train,
    y_train,
    train_size=6000,
    random_state=42,
    stratify=y_train
)

np.savez_compressed(
    "data/fair_train_subset.npz",
    X_train=X_small,
    y_train=y_small
)

print("Fair training subset saved!")
print("Training samples:", len(X_small))
print("Samples per class:", np.bincount(y_small))
