"""
Merges Person A's and Person B's results into one table for the report.

  python combine_results.py
  python combine_results.py --a combined_model_results.csv --b results/cnn_model_results.csv

Writes all_model_results.csv: every model, grouped by number of classes,
best test accuracy first. Both inputs use the same columns.
"""
import argparse
import csv

COLUMNS = ["Model", "Validation Accuracy (%)", "Test Accuracy (%)",
           "Training Time (s)", "Training Samples", "Classes"]


def read_rows(path, owner):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in COLUMNS if rows and c not in rows[0]]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    out = []
    for r in rows:
        out.append({
            "Model": r["Model"],
            # rounding also cleans Person A's long decimals like 50.324000000000005
            "Validation Accuracy (%)": round(float(r["Validation Accuracy (%)"]), 2),
            "Test Accuracy (%)": round(float(r["Test Accuracy (%)"]), 2),
            "Training Time (s)": round(float(r["Training Time (s)"]), 2),
            "Training Samples": int(r["Training Samples"]),
            "Classes": int(r["Classes"]),
            "Person": owner,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="combined_model_results.csv", help="Person A's results")
    p.add_argument("--b", default="results/cnn_model_results.csv", help="Person B's results")
    p.add_argument("--out", default="all_model_results.csv")
    args = p.parse_args()

    rows = read_rows(args.a, "A") + read_rows(args.b, "B")
    rows.sort(key=lambda r: (r["Classes"], -r["Test Accuracy (%)"]))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + ["Person"])
        w.writeheader()
        w.writerows(rows)

    # quick printout grouped by class count
    for k in sorted({r["Classes"] for r in rows}):
        print(f"\n{k} classes")
        print(f"  {'Model':<28}{'Val %':>8}{'Test %':>8}{'Time s':>10}{'Samples':>9}")
        for r in (r for r in rows if r["Classes"] == k):
            print(f"  {r['Model']:<28}{r['Validation Accuracy (%)']:>8}"
                  f"{r['Test Accuracy (%)']:>8}{r['Training Time (s)']:>10}"
                  f"{r['Training Samples']:>9}")
    print(f"\nSaved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()