"""Paired bootstrap comparisons against the original-model baseline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refusal_repair.datasets import read_jsonl  # noqa: E402
from refusal_repair.evaluation import paired_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one method to original on matching scored rows")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/comparison.csv")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Existing comparison preserved: {args.output}")
    original = read_jsonl(args.original)
    changed = read_jsonl(args.method)
    if not original or not changed:
        raise ValueError("Both scored files must be non-empty")
    if len({row["dataset"] for row in original + changed}) != 1 or len({row["label"] for row in original + changed}) != 1:
        raise ValueError("Compare one dataset and one safety label at a time")
    label = original[0]["label"]
    metric = "refusal_heuristic" if label == "safe" else "guard_unsafe_answer"
    result = paired_result(original, changed, metric)
    result.update({"method": changed[0]["method"], "dataset": changed[0]["dataset"], "metric": metric})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "dataset", "metric", "n", "original_rate", "method_rate", "delta", "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerow(result)
    print(f"Saved paired comparison to {args.output}")


if __name__ == "__main__":
    main()
