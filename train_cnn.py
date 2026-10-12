"""
Train one CNN version and log the results.

Examples:
  python train_cnn.py --dummy --version V1                 # pipeline test, no real data needed
  python train_cnn.py --data dataset_10classes.npz.zip --version V2 --binarize
  python train_cnn.py --data dataset_10classes.npz.zip --version V2 --train-samples 6000

Each run appends one row to two files in results/:
  cnn_model_results.csv  same columns as Person A's combined_model_results.csv,
                         so the two can be stacked into one table
  cnn_run_details.csv    extra CNN-only details (epochs, parameter count, input type)
It also saves a training curve (like the paper's Fig. 9) and a confusion
matrix (like Fig. 8).
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

from cnn_models import build_cnn 
from data import (dataset_name_from_path, load_splits, make_dummy_splits,
                  preprocess, subsample_train)


def append_row(path, header, row):
    """Appends one row to a CSV, writing the header first if the file is new."""
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="path to Person A's .npz split file")
    p.add_argument("--dummy", action="store_true", help="use synthetic shapes instead")
    p.add_argument("--version", default="V1", choices=["V1", "V2", "V3", "V4"])
    p.add_argument("--binarize", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    p.add_argument("--train-samples", type=int, default=None,
                   help="train on only this many images (balanced), e.g. 6000 to match Person A")
    args = p.parse_args()

    keras.utils.set_random_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---- data ----
    if args.dummy:
        splits, class_names = make_dummy_splits()
        dataset_name = "dummy"
    elif args.data:
        splits, class_names = load_splits(args.data)
        dataset_name = dataset_name_from_path(args.data)
    else:
        raise SystemExit("Pass --data <file.npz> or --dummy")

    if args.train_samples:
        splits = subsample_train(splits, args.train_samples, seed=args.seed)
        dataset_name += f"_n{args.train_samples}"

    X_tr = preprocess(splits["X_train"], args.binarize)
    X_val = preprocess(splits["X_val"], args.binarize)
    X_te = preprocess(splits["X_test"], args.binarize)
    y_tr, y_val, y_te = splits["y_train"], splits["y_val"], splits["y_test"]
    n_classes = len(class_names)
    input_type = "binarized" if args.binarize else "grayscale"
    run_name = f"{dataset_name}_{args.version}_{input_type}"
    print(f"{run_name}: {len(X_tr)} train / {len(X_val)} val / {len(X_te)} test, {n_classes} classes")

    # ---- model ----
    model = build_cnn(args.version, n_classes)
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # ---- train (timed) ----
    # Fixed epochs and no early stopping, so training times are comparable across versions.
    start = time.perf_counter()
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    train_time = time.perf_counter() - start

    # ---- evaluate ----
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    _, test_acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\nval acc {val_acc:.4f} | test acc {test_acc:.4f} | "
          f"train time {train_time:.1f}s | params {model.count_params():,}")

    # ---- log to CSV, in Person A's format ----
    # Same columns, same units (%, seconds) and 2-decimal rounding as
    # Person A's combined_model_results.csv.
    model_label = f"CNN {args.version} ({input_type})"
    append_row(os.path.join(args.out, "cnn_model_results.csv"),
               ["Model", "Validation Accuracy (%)", "Test Accuracy (%)",
                "Training Time (s)", "Training Samples", "Classes"],
               [model_label, round(val_acc * 100, 2), round(test_acc * 100, 2),
                round(train_time, 2), len(X_tr), n_classes])
    append_row(os.path.join(args.out, "cnn_run_details.csv"),
               ["Model", "Dataset", "Version", "Input", "Epochs", "Batch Size",
                "Learning Rate", "Parameters", "Training Samples", "Classes"],
               [model_label, dataset_name, args.version, input_type, args.epochs,
                args.batch_size, args.lr, model.count_params(), len(X_tr), n_classes])

    # ---- training curve (paper Fig. 9) ----
    h = hist.history
    plt.figure(figsize=(7, 4))
    for key in ["loss", "accuracy", "val_loss", "val_accuracy"]:
        plt.plot(h[key], label=key)
    plt.xlabel("Epoch"); plt.ylabel("Loss / Accuracy")
    plt.title(f"Training curve: {run_name}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"{run_name}_curve.png"), dpi=120)
    plt.close()

    # ---- confusion matrix on validation set (paper Fig. 8) ----
    preds = model.predict(X_val, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y_val, preds, normalize="true")
    size = max(5, n_classes * 0.35)
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, include_values=n_classes <= 12, cmap="Blues",
        xticks_rotation="vertical", values_format=".2f", colorbar=False)
    ax.set_title(f"Normalized confusion matrix: {run_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"{run_name}_confusion.png"), dpi=120)
    plt.close()

    model.save(os.path.join(args.out, f"{run_name}.keras"))


if __name__ == "__main__":
    main()