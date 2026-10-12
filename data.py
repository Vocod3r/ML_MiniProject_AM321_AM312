"""
Data loading for the CNN experiments.

DATA FORMAT (Person A's files)
One file per class count: dataset_3classes.npz.zip, dataset_10classes.npz.zip,
dataset_50classes.npz.zip (np.load opens them directly; .npz is a zip format). Each holds:
    X_train, X_val, X_test : float32, shape (N, 784), already scaled to 0-1
    y_train, y_val, y_test : int64 class indices 0..K-1
    classes                : class names, length K, in label order
Split is 80/10/10 with balanced classes (16,000 / 2,000 / 2,000 per class).
load_splits() also accepts raw 0-255 images and a "class_names" key.
Use make_dummy_splits() to test the pipeline without real data.
"""
import numpy as np


def load_splits(path):
    """Loads a split file. Works with .npz and with .npz.zip (it's the same format).

    Handles two differences in Person A's files:
      - class names are stored under "classes" instead of "class_names"
      - images are float32 already scaled to 0-1 instead of raw 0-255
    Images are converted back to 0-255 so preprocess() treats every file the same way.
    """
    d = np.load(path, allow_pickle=True)
    splits = {k: d[k] for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]}

    name_key = "class_names" if "class_names" in d.files else "classes"
    class_names = [str(c) for c in d[name_key]]

    for k in ["X_train", "X_val", "X_test"]:
        X = splits[k]
        if X.max() <= 1.0:  # already scaled to 0-1 -> back to 0-255
            X = np.rint(X * 255)
        splits[k] = X.astype(np.uint8)
    return splits, class_names


def dataset_name_from_path(path):
    """'data/dataset_10classes.npz.zip' -> 'dataset_10classes'"""
    import os
    name = os.path.basename(path)
    for ext in [".zip", ".npz"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name


def subsample_train(splits, n_total, seed=0):
    """Keeps n_total training images, the same number from each class.
    Val and test are untouched. Used to match Person A's 6000-sample runs."""
    rng = np.random.default_rng(seed)
    y = splits["y_train"]
    classes = np.unique(y)
    per_class = n_total // len(classes)
    keep = np.concatenate([rng.choice(np.where(y == c)[0], per_class, replace=False)
                           for c in classes])
    keep = rng.permutation(keep)
    out = dict(splits)
    out["X_train"], out["y_train"] = splits["X_train"][keep], y[keep]
    return out


def preprocess(X, binarize=False, threshold=127):
    """Reshape to (N, 28, 28, 1) floats.
    Grayscale -> scaled to [0, 1]. Binarized -> every pixel is exactly 0 or 1.
    (The paper doesn't give its threshold; 127 is our choice, so say so in the report.)
    """
    X = X.reshape(-1, 28, 28, 1)
    if binarize:
        return (X > threshold).astype("float32")
    return X.astype("float32") / 255.0


def make_dummy_splits(n_per_class=1000, seed=0):
    """Synthetic 'doodles' (lines, circles, squares, crosses) with random
    position, size and thickness. Unlike pure noise, a working CNN should
    learn these, so it checks the whole pipeline end to end."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:28, 0:28]
    class_names = ["hline", "vline", "circle", "square", "cross"]

    def draw(kind):
        img = np.zeros((28, 28), dtype=np.float32)
        cx, cy = rng.integers(9, 19, size=2)
        r = rng.integers(4, 9)
        t = rng.uniform(0.8, 1.8)  # stroke thickness
        if kind == "hline":
            img[(np.abs(yy - cy) < t) & (np.abs(xx - cx) < r)] = 1
        elif kind == "vline":
            img[(np.abs(xx - cx) < t) & (np.abs(yy - cy) < r)] = 1
        elif kind == "circle":
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            img[np.abs(dist - r) < t] = 1
        elif kind == "square":
            inside = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
            border = (np.abs(np.abs(xx - cx) - r) < t) | (np.abs(np.abs(yy - cy) - r) < t)
            img[inside & border] = 1
        elif kind == "cross":
            img[(np.abs(yy - cy) < t) & (np.abs(xx - cx) < r)] = 1
            img[(np.abs(xx - cx) < t) & (np.abs(yy - cy) < r)] = 1
        img += rng.normal(0, 0.1, img.shape)  # a bit of noise
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    X, y = [], []
    for label, kind in enumerate(class_names):
        for _ in range(n_per_class):
            X.append(draw(kind))
            y.append(label)
    X, y = np.array(X), np.array(y)

    # same 80/10/10 split as the paper
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_tr, n_val = int(0.8 * len(X)), int(0.1 * len(X))
    splits = {
        "X_train": X[:n_tr], "y_train": y[:n_tr],
        "X_val": X[n_tr:n_tr + n_val], "y_val": y[n_tr:n_tr + n_val],
        "X_test": X[n_tr + n_val:], "y_test": y[n_tr + n_val:],
    }
    return splits, class_names