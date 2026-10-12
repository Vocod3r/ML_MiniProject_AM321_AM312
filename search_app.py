"""
Draw-to-search demo: draw a doodle in the browser, the CNN names it, and the app
finds the most similar doodles among every image in the test set.

  pip install flask
  python search_app.py
  python search_app.py --model results/dataset_10classes_n6000_V1_grayscale.keras --data dataset_10classes.npz.zip

Then open http://127.0.0.1:5000 in a browser.

How the search works
  1. The model's last hidden layer (Dense 128) turns any doodle into 128 numbers,
     a "fingerprint". At start-up we compute the fingerprint of every test image.
  2. Your drawing is redrawn the way QuickDraw drawings were made (strokes scaled into
     a 256 x 256 box, even line width, shrunk to 28 x 28), then classified and fingerprinted.
  3. Cosine similarity between your fingerprint and all test fingerprints gives the
     closest doodles. Their labels are a second opinion on the model's guess.
"""
import argparse
import base64
import glob
import io
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageDraw
from tensorflow import keras

from data import load_splits, preprocess

HERE = os.path.dirname(os.path.abspath(__file__))
INK = np.array([27, 42, 65], dtype=np.float32)        # navy, matches the page
PAPER = np.array([255, 255, 255], dtype=np.float32)


# ---------------------------------------------------------------- setup
def find_default(pattern):
    hits = sorted(glob.glob(os.path.join(HERE, pattern)))
    return hits[0] if hits else None


def parse_args():
    p = argparse.ArgumentParser(description="Draw-to-search demo")
    p.add_argument("--model", default=find_default("results/*10classes*V1_grayscale.keras"),
                   help="trained .keras file from train_cnn.py")
    p.add_argument("--data", default=find_default("dataset_10classes.npz*") or
                   find_default("data/dataset_10classes.npz*"),
                   help="split file whose test images are searched")
    p.add_argument("--split", default="X_test", choices=["X_test", "X_val", "X_train"],
                   help="which images to search (default: the test set)")
    p.add_argument("--neighbours", type=int, default=12)
    p.add_argument("--port", type=int, default=5000)
    return p.parse_args()


class SearchIndex:
    def __init__(self, model_path, data_path, split, k):
        t0 = time.perf_counter()
        self.model = keras.models.load_model(model_path)
        self.binarize = "binarized" in os.path.basename(model_path)
        # A second model that stops at the last hidden layer = the fingerprint maker.
        self.embedder = keras.Model(self.model.inputs, self.model.layers[-2].output)

        splits, self.classes = load_splits(data_path)
        self.images = splits[split]                                # uint8, 0-255
        self.labels = splits["y" + split[1:]]
        if len(self.classes) != self.model.output_shape[-1]:
            raise SystemExit(f"The model predicts {self.model.output_shape[-1]} classes but "
                             f"{os.path.basename(data_path)} has {len(self.classes)}. "
                             "Use a model trained on the same data file.")

        X = preprocess(self.images, self.binarize)
        emb = self.embedder.predict(X, batch_size=512, verbose=0)
        self.emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        self.k = k
        self.model_name = os.path.basename(model_path)
        self.data_name = os.path.basename(data_path)
        print(f"Indexed {len(self.images):,} doodles in {time.perf_counter() - t0:.1f}s "
              f"({self.model_name}, {len(self.classes)} classes)")

    def search(self, img28):
        x = preprocess(img28[None], self.binarize)
        probs = self.model.predict(x, verbose=0)[0]
        q = self.embedder.predict(x, verbose=0)[0]
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = self.emb @ q
        nearest = np.argsort(-sims)[: self.k]

        order = np.argsort(-probs)
        votes = np.bincount(self.labels[nearest], minlength=len(self.classes))
        top_vote = int(votes.argmax())
        return {
            "guesses": [{"label": self.classes[i], "prob": float(probs[i])} for i in order[:3]],
            "neighbours": [{"label": self.classes[self.labels[i]], "similarity": float(sims[i]),
                            "image": to_png(self.images[i])} for i in nearest],
            "vote": {"label": self.classes[top_vote], "count": int(votes[top_vote]), "of": self.k},
            "seen": to_png(img28),
        }


# ---------------------------------------------------------------- drawing -> 28 x 28
def strokes_to_28(strokes, box=256, pad=12, width=14):
    """Redraws pen strokes QuickDraw-style: white ink on black, scaled to fill a
    256 x 256 box with even line width, then shrunk to 28 x 28."""
    pts = [np.asarray(s, dtype=np.float32).reshape(-1, 2) for s in strokes if len(s)]
    if not pts:
        return None
    allp = np.concatenate(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = max(float((hi - lo).max()), 1.0)
    scale = (box - 2 * pad) / span
    offset = (box - (hi - lo) * scale) / 2              # centre the drawing

    big = Image.new("L", (box, box), 0)
    draw = ImageDraw.Draw(big)
    r = width / 2
    for s in pts:
        xy = [tuple(p) for p in ((s - lo) * scale + offset)]
        if len(xy) == 1:
            x, y = xy[0]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        else:
            draw.line(xy, fill=255, width=width, joint="curve")
            for x, y in (xy[0], xy[-1]):                 # round the line ends
                draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
    small = big.resize((28, 28), Image.LANCZOS)
    return np.asarray(small, dtype=np.uint8)


def to_png(img):
    """28 x 28 doodle (white on black, 0-255) -> navy-on-white PNG data URL."""
    a = np.asarray(img, dtype=np.float32).reshape(28, 28, 1) / 255.0
    rgb = (PAPER * (1 - a) + INK * a).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- web app
def create_app(index):
    app = Flask(__name__)

    @app.get("/")
    def page():
        return send_file(os.path.join(HERE, "search_app.html"))

    @app.get("/api/info")
    def info():
        return jsonify({"classes": index.classes, "count": len(index.images),
                        "model": index.model_name, "data": index.data_name})

    @app.post("/api/search")
    def search():
        body = request.get_json(silent=True) or {}
        img = strokes_to_28(body.get("strokes", []))
        if img is None:
            return jsonify({"error": "Draw something first."}), 400
        return jsonify(index.search(img))

    return app


if __name__ == "__main__":
    args = parse_args()
    if not args.model or not os.path.exists(args.model):
        raise SystemExit("No model found. Train one first, for example:\n"
                         "  python train_cnn.py --data dataset_10classes.npz.zip --version V1 --train-samples 6000\n"
                         "or pass --model path/to/model.keras")
    if not args.data or not os.path.exists(args.data):
        raise SystemExit("No data file found. Put dataset_10classes.npz.zip in this folder "
                         "or pass --data path/to/file")
    idx = SearchIndex(args.model, args.data, args.split, args.neighbours)
    print(f"Open http://127.0.0.1:{args.port} in your browser (Ctrl + C to stop).")
    create_app(idx).run(host="127.0.0.1", port=args.port, debug=False)