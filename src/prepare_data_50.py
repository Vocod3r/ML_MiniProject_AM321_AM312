import os
import urllib.request
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
SEED = 42
SAMPLES_PER_CLASS = 20000

CLASSES = [
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

os.makedirs(DATA_DIR, exist_ok=True)


def download_class(name):
    filename = name.replace(" ", "_") + ".npy"
    path = os.path.join(DATA_DIR, filename)
    url = BASE_URL + name.replace(" ", "%20") + ".npy"

    if os.path.exists(path) and os.path.getsize(path) > 1000000:
        try:
            data = np.load(path, mmap_mode="r")
            if len(data) >= SAMPLES_PER_CLASS:
                return data
        except Exception:
            pass

    for attempt in range(1, 6):
        try:
            print(f"Downloading {name} (attempt {attempt}/5)...")
            temp_path = path + ".download"

            urllib.request.urlretrieve(url, temp_path)
            os.replace(temp_path, path)

            data = np.load(path, mmap_mode="r")
            return data

        except Exception as e:
            print(f"Download failed: {e}")
            for candidate in [path + ".download"]:
                if os.path.exists(candidate):
                    os.remove(candidate)

            if attempt < 5:
                import time
                time.sleep(2 * attempt)

    raise RuntimeError(f"Could not download {name} after 5 attempts.")

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
        os.path.join(DATA_DIR, "dataset_50classes.npz"),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        classes=np.array(CLASSES)
    )

    print("\n50-class dataset ready!")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

if __name__ == "__main__":
    main()


