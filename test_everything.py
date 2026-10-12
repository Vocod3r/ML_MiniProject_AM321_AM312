"""
Tests everything in one go:   python test_everything.py

  1. Packages      are tensorflow, numpy, scikit-learn and matplotlib installed?
  2. Code tests    the 15 tests in test_pipeline.py
  3. Data files    checks every dataset_*classes.npz(.zip) file in this folder
  4. Real run      a quick 1-epoch training run on the smallest data file
  5. Merge         combines Person A's and Person B's results, if both exist

Takes about 2-3 minutes. Nothing in your results/ folder is changed.
"""
import glob
import importlib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
summary = []


def step(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


def record(name, ok, note=""):
    summary.append((name, ok, note))
    shown = f"  -> {note}" if note and not ok else ""
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{shown}", flush=True)


def run(cmd, show=True):
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, env=env)
    if show:
        for line in r.stdout.splitlines():
            if line.strip().startswith(("PASS", "FAIL", "->")) or "tests passed" in line:
                print("  " + line.strip())
    return r


# ---------------------------------------------------------------- 1
step("1. Packages")
missing = []
for pkg in ["tensorflow", "numpy", "sklearn", "matplotlib"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)
record("required packages installed", not missing,
       f"missing {missing}; run: pip install tensorflow numpy scikit-learn matplotlib"
       if missing else "")
if missing:
    sys.exit("\nInstall the missing packages, then run this again.")

needed = ["data.py", "cnn_models.py", "train_cnn.py", "test_pipeline.py",
          "run_sweep.py", "combine_results.py"]
absent = [f for f in needed if not os.path.exists(f) or os.path.getsize(f) == 0]
record("all code files present and not empty", not absent, f"missing or empty: {absent}")
if absent:
    sys.exit("\nPaste in the missing files, then run this again.")

# ---------------------------------------------------------------- 2
step("2. Code tests (test_pipeline.py)")
r = run(["test_pipeline.py"])
record("all code tests pass", r.returncode == 0,
       "" if r.returncode == 0 else (r.stderr.strip().splitlines() or ["see above"])[-1])

# ---------------------------------------------------------------- 3
step("3. Data files from Person A")
data_files = sorted(set(glob.glob("dataset_*classes.npz*") + glob.glob("data/dataset_*classes.npz*")),
                    key=os.path.getsize)
data_files = [f for f in data_files if os.path.getsize(f) > 0 and not f.endswith(".crswap")]
if not data_files:
    record("data files found", False, "no dataset_*classes.npz files in this folder")
for f in data_files:
    print(f"\n  {f}")
    r = run(["test_pipeline.py", "--check", f])
    record(f"{os.path.basename(f)} is valid", r.returncode == 0)

# ---------------------------------------------------------------- 4
step("4. Quick real training run (1 epoch, 1500 images)")
if data_files:
    with tempfile.TemporaryDirectory() as tmp:
        r = run(["train_cnn.py", "--data", data_files[0], "--version", "V2",
                 "--epochs", "1", "--train-samples", "1500", "--out", tmp], show=False)
        acc_line = next((l for l in r.stdout.splitlines() if "val acc" in l), "")
        print("  " + acc_line.strip())
        ok = r.returncode == 0 and os.path.exists(os.path.join(tmp, "cnn_model_results.csv"))
        record("training runs on real data", ok,
               "" if ok else (r.stderr.strip().splitlines() or ["unknown error"])[-1])
        if ok:
            with open(os.path.join(tmp, "cnn_model_results.csv")) as fh:
                header = fh.readline().strip()
            expected = ("Model,Validation Accuracy (%),Test Accuracy (%),"
                        "Training Time (s),Training Samples,Classes")
            record("results use Person A's columns", header == expected, header)
else:
    record("training runs on real data", False, "skipped: no data files")

# ---------------------------------------------------------------- 5
step("5. Merging results")
a, b = "combined_model_results.csv", os.path.join("results", "cnn_model_results.csv")
if os.path.exists(a) and os.path.exists(b):
    with tempfile.TemporaryDirectory() as tmp:
        r = run(["combine_results.py", "--a", a, "--b", b,
                 "--out", os.path.join(tmp, "merged.csv")], show=False)
        record("Person A + Person B results merge", r.returncode == 0,
               (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0
               else (r.stderr.strip().splitlines() or ["error"])[-1])
else:
    have = [f for f in (a, b) if os.path.exists(f)]
    print(f"  SKIP  needs both {a} and {b} (found: {have or 'neither'}).")
    print("        Run run_sweep.py first to create Person B's results.")

# ---------------------------------------------------------------- summary
step("SUMMARY")
for name, ok, _ in summary:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
failed = sum(not ok for _, ok, _ in summary)
print(f"\n{len(summary) - failed}/{len(summary)} checks passed"
      + ("  -- everything works." if not failed else "  -- see the FAIL lines above."))
sys.exit(1 if failed else 0)