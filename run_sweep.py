"""
Runs every CNN version on grayscale and binarized input for each data file.
Works on Windows, Mac, Linux and Colab (unlike run_sweep.sh).

  python run_sweep.py dataset_3classes.npz.zip dataset_10classes.npz.zip
  python run_sweep.py dataset_3classes.npz.zip --train-samples 6000   # match Person A's sample size
  python run_sweep.py --dummy                                          # quick test

All results go into results/cnn_model_results.csv.
"""
import subprocess
import sys

args = sys.argv[1:]
extra = []
if "--train-samples" in args:
    i = args.index("--train-samples")
    extra = ["--train-samples", args[i + 1]]
    del args[i:i + 2]
sources = args or ["--dummy"]

runs = []
for src in sources:
    data_arg = ["--dummy"] if src == "--dummy" else ["--data", src]
    for v in ["V1", "V2", "V3", "V4"]:
        runs.append(data_arg + ["--version", v] + extra)
        runs.append(data_arg + ["--version", v, "--binarize"] + extra)

for n, run in enumerate(runs, 1):
    print(f"\n===== Run {n}/{len(runs)}: {' '.join(run)} =====", flush=True)
    result = subprocess.run([sys.executable, "train_cnn.py"] + run)
    if result.returncode != 0:
        sys.exit(f"Run {n} failed, stopping. Fix the error above and rerun.")

print("\nDone. Results in results/cnn_model_results.csv")