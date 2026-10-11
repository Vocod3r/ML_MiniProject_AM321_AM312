"""
Tests for Person B's code. No extra packages needed.

  python test_pipeline.py                      # run all tests (~1 minute on a laptop)
  python test_pipeline.py --check dataset_10classes.npz.zip  # check a file from Person A before training on it
"""
import argparse
import os
import subprocess
import sys
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide TensorFlow's startup messages
import numpy as np

from cnn_models import build_cnn
from data import (dataset_name_from_path, load_splits, make_dummy_splits,
                  preprocess, subsample_train)

results = []


def test(name):
    """Decorator: runs a test, prints PASS or FAIL, and keeps going either way."""
    def wrap(fn):
        try:
            fn()
            print(f"  PASS  {name}")
            results.append(True)
        except Exception as e:
            print(f"  FAIL  {name}\n        -> {type(e).__name__}: {e}")
            results.append(False)
        return fn
    return wrap


def run_all_tests():
    print("\n[1] Data")

    @test("dummy data has the right shapes and an 80/10/10 split")
    def _():
        s, names = make_dummy_splits(n_per_class=100)
        total = 100 * len(names)
        assert s["X_train"].shape == (int(0.8 * total), 28, 28)
        assert len(s["X_val"]) == int(0.1 * total)
        assert len(s["X_train"]) + len(s["X_val"]) + len(s["X_test"]) == total
        assert s["X_train"].dtype == np.uint8

    @test("every class appears in train, val and test")
    def _():
        s, names = make_dummy_splits(n_per_class=100)
        for key in ["y_train", "y_val", "y_test"]:
            assert set(s[key]) == set(range(len(names))), f"{key} is missing a class"

    @test("grayscale preprocessing gives shape (N,28,28,1) with values in [0,1]")
    def _():
        X = np.random.randint(0, 256, size=(10, 784), dtype=np.uint8)
        out = preprocess(X)
        assert out.shape == (10, 28, 28, 1)
        assert out.min() >= 0 and out.max() <= 1

    @test("binarized preprocessing gives only 0s and 1s")
    def _():
        X = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
        out = preprocess(X, binarize=True)
        assert set(np.unique(out)) <= {0.0, 1.0}

    @test("save and load round trip works (the format Person A will use)")
    def _():
        s, names = make_dummy_splits(n_per_class=20)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fake.npz")
            np.savez(path, class_names=np.array(names), **s)
            loaded, loaded_names = load_splits(path)
            assert loaded_names == names
            assert np.array_equal(loaded["X_test"], s["X_test"])

    @test("Person A's format loads correctly (key 'classes', floats 0-1, .npz.zip name)")
    def _():
        s, names = make_dummy_splits(n_per_class=20)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dataset_5classes.npz.zip")
            scaled = {k: (v.astype("float32") / 255 if k.startswith("X") else v)
                      for k, v in s.items()}
            with open(path, "wb") as fh:
                np.savez(fh, classes=np.array(names), **scaled)
            loaded, loaded_names = load_splits(path)
            assert loaded_names == names
            assert loaded["X_val"].dtype == np.uint8 and loaded["X_val"].max() > 200
            assert np.abs(loaded["X_val"].astype(int) - s["X_val"]).max() <= 1
            assert dataset_name_from_path(path) == "dataset_5classes"

    @test("training subsample keeps the right size and stays balanced")
    def _():
        s, names = make_dummy_splits(n_per_class=100)
        sub = subsample_train(s, 250)
        assert len(sub["y_train"]) == 250
        assert set(np.bincount(sub["y_train"])) == {50}
        assert len(sub["y_val"]) == len(s["y_val"]), "val set must not change"

    print("\n[2] Models")

    for v in ["V1", "V2", "V3", "V4"]:
        @test(f"{v} builds and outputs one probability per class")
        def _(v=v):
            m = build_cnn(v, n_classes=7)
            out = m.predict(np.zeros((3, 28, 28, 1), "float32"), verbose=0)
            assert out.shape == (3, 7)
            assert np.allclose(out.sum(axis=1), 1, atol=1e-5), "softmax rows should sum to 1"

    @test("each version matches its description (layers and parameter counts)")
    def _():
        expected = {"V1": 421_642, "V2": 804_554, "V3": 3_213_002, "V4": 1_606_666}
        for v, n in expected.items():
            got = build_cnn(v, n_classes=10).count_params()
            assert got == n, f"{v}: expected {n:,} params, got {got:,}"

    @test("an unknown version name raises an error")
    def _():
        try:
            build_cnn("V9", 5)
        except ValueError:
            return
        raise AssertionError("build_cnn('V9') should have raised ValueError")

    print("\n[3] Learning")

    @test("V2 learns the dummy shapes (well above chance after 2 epochs)")
    def _():
        s, names = make_dummy_splits(n_per_class=300)
        m = build_cnn("V2", len(names))
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(preprocess(s["X_train"]), s["y_train"], epochs=2, batch_size=64, verbose=0)
        _, acc = m.evaluate(preprocess(s["X_val"]), s["y_val"], verbose=0)
        chance = 1 / len(names)
        assert acc > 3 * chance, f"accuracy {acc:.2f} is too close to chance ({chance:.2f})"

    print("\n[4] Full script (train_cnn.py end to end)")

    @test("train_cnn.py runs and writes results in Person A's format, plots and model")
    def _():
        with tempfile.TemporaryDirectory() as d:
            cmd = [sys.executable, "train_cnn.py", "--dummy", "--version", "V2",
                   "--epochs", "1", "--out", d]
            r = subprocess.run(cmd, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-500:]
            files = os.listdir(d)
            for f in ["cnn_model_results.csv", "cnn_run_details.csv",
                      "dummy_V2_grayscale_curve.png", "dummy_V2_grayscale_confusion.png",
                      "dummy_V2_grayscale.keras"]:
                assert f in files, f"missing {f}"
            with open(os.path.join(d, "cnn_model_results.csv")) as fh:
                lines = fh.read().splitlines()
            assert len(lines) == 2, "CSV should have a header and one row"
            person_a_header = ("Model,Validation Accuracy (%),Test Accuracy (%),"
                               "Training Time (s),Training Samples,Classes")
            assert lines[0] == person_a_header, f"header differs from Person A's: {lines[0]}"
            assert lines[1].startswith("CNN V2 (grayscale),"), lines[1]

    passed = sum(results)
    print(f"\n{passed}/{len(results)} tests passed")
    return passed == len(results)


def check_split_file(path):
    """Checks a split file from Person A before you train on it.
    Accepts .npz or .npz.zip, "classes" or "class_names", and 0-1 or 0-255 images."""
    print(f"\nChecking {path}")
    d = np.load(path, allow_pickle=True)

    @test("all 6 arrays plus class names are present")
    def _():
        missing = [k for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
                   if k not in d.files]
        if "class_names" not in d.files and "classes" not in d.files:
            missing.append("class_names (or classes)")
        assert not missing, f"missing: {missing}"

    splits, names = load_splits(path)

    @test("images are 28x28 grayscale (not binarized)")
    def _():
        for k in ["X_train", "X_val", "X_test"]:
            X = splits[k]
            assert X[0].size == 784, f"{k}: each image should have 784 pixels"
            assert len(np.unique(X[:200])) > 2, f"{k}: looks binarized; we need grayscale"

    @test("labels are whole numbers 0..K-1 and match the class names")
    def _():
        K = len(names)
        for k in ["y_train", "y_val", "y_test"]:
            y = splits[k]
            assert np.issubdtype(y.dtype, np.integer), f"{k} should be integers"
            assert y.min() == 0 and y.max() == K - 1, f"{k} should run 0..{K - 1}"

    @test("image and label counts match in each split")
    def _():
        for s in ["train", "val", "test"]:
            assert len(splits[f"X_{s}"]) == len(splits[f"y_{s}"]), f"{s}: counts differ"

    @test("split is roughly 80/10/10")
    def _():
        n = [len(splits[f"y_{s}"]) for s in ["train", "val", "test"]]
        frac = np.array(n) / sum(n)
        assert np.allclose(frac, [0.8, 0.1, 0.1], atol=0.02), f"got {np.round(frac, 3)}"

    @test("classes are balanced in the training set")
    def _():
        counts = np.bincount(splits["y_train"])
        assert counts.min() > 0.9 * counts.max(), f"counts range {counts.min()}-{counts.max()}"

    @test("preprocessing gives sensible images (not all black)")
    def _():
        g = preprocess(splits["X_val"][:200])
        b = preprocess(splits["X_val"][:200], binarize=True)
        assert 0.02 < g.mean() < 0.6, f"grayscale mean {g.mean():.3f} looks wrong"
        assert 0.02 < b.mean() < 0.6, f"binarized mean {b.mean():.3f} looks wrong"

    n = [len(splits[f"y_{s}"]) for s in ["train", "val", "test"]]
    print(f"\n  {len(names)} classes, train/val/test = {n[0]}/{n[1]}/{n[2]}")
    print(f"  classes: {names}")
    return all(results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", help="a split file from Person A to check")
    args = p.parse_args()
    ok = check_split_file(args.check) if args.check else run_all_tests()
    sys.exit(0 if ok else 1)