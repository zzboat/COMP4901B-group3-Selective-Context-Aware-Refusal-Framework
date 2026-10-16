"""Paired bootstrap comparisons against the original-model baseline."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refusal_repair.datasets import read_jsonl  # noqa: E402


def paired_result(original: list[dict], changed: list[dict], metric: str, seed: int = 42) -> dict:
    base = {row["id"]: row for row in original}
    variant = {row["id"]: row for row in changed}
    if len(base) != len(original) or len(variant) != len(changed) or set(base) != set(variant):
        raise ValueError("Paired comparison requires the same unique prompt IDs")
    ids = sorted(base)
    differences = []
    for key in ids:
        if metric not in base[key] or metric not in variant[key]:
            raise ValueError(f"Missing {metric} label for {key}")
        differences.append(float(variant[key][metric]) - float(base[key][metric]))
    rng = random.Random(seed)
    draws = [sum(rng.choices(differences, k=len(ids))) / len(ids) for _ in range(2000)]
    draws.sort()
    return {
        "n": len(ids),
        "original_rate": sum(float(base[key][metric]) for key in ids) / len(ids),
        "method_rate": sum(float(variant[key][metric]) for key in ids) / len(ids),
        "delta": sum(differences) / len(ids),
        "ci_low": draws[49],
        "ci_high": draws[1949],
    }


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
