from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refusal_repair.score_files import load_scored  # noqa: E402
from refusal_repair.evaluation import summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate scored baseline files")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/baselines.csv")
    args = parser.parse_args()
    rows = [row for path in args.inputs for row in load_scored(ROOT, path)]
    table = summarize(rows)
    if args.output.exists():
        raise FileExistsError(f"Existing table preserved: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["method", "dataset", "label", "n", "refusal_rate", "guard_unsafe_answer_rate", "mean_model_calls", "mean_elapsed_sec"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved {len(table)} summary rows to {args.output}")


if __name__ == "__main__":
    main()
