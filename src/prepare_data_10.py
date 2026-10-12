import os
import urllib.request
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
SEED = 42
SAMPLES_PER_CLASS = 20000

CLASSES = [
    "banana",
    "apple",
    "cat",
    "dog",
    "car",
    "bicycle",
    "airplane",
    "chair",
    "tree",
    "fish"
]

os.makedirs(DATA_DIR, exist_ok=True)

def download_class(name):
    filename = name.replace(" ", "%20")
    url = (
        "https://storage.googleapis.com/quickdraw_dataset/"
        f"full/numpy_bitmap/{filename}.npy"
    )

    path = os.path.join(DATA_DIR, f"{name.replace(' ', '_')}.npy")

    if not os.path.exists(path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)

    return np.load(path, mmap_mode="r")

def main():
    images, labels = [], []

    for label, name in enumerate(CLASSES):
        print(f"Preparing {name}...")
        data = download_class(name)

        rng = np.random.default_rng(SEED)
        indices = rng.permutation(len(data))[:SAMPLES_PER_CLASS]

        images.append(np.asarray(data[indices]))
        labels.extend([label] * len(indices))

    X = np.concatenate(images).astype(np.float32) / 255.0
    y = np.array(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    np.savez_compressed(
        os.path.join(DATA_DIR, "dataset_10classes.npz"),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        classes=np.array(CLASSES)
    )

    print("\n10-class dataset ready!")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)
    print("Classes:", CLASSES)

if __name__ == "__main__":
    main()
