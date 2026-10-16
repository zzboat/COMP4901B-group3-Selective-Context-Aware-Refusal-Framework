from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refusal_repair.datasets import read_jsonl, write_jsonl  # noqa: E402
from refusal_repair.evaluation import LlamaGuardJudge, label_response  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Label one generated-response file")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--guard", action="store_true", help="Also classify answer safety with Llama Guard 3 1B")
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".scored.jsonl")
    if output.exists():
        raise FileExistsError(f"Existing scores preserved: {output}")
    rows = read_jsonl(args.input)
    judge = None
    if args.guard:
        config = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
        judge = LlamaGuardJudge(ROOT / config["judge_path"])
    scored = [
        label_response(row, judge.unsafe_answer(row["prompt"], row["response"]) if judge else None)
        for row in rows
    ]
    write_jsonl(output, scored)
    print(f"Saved {len(scored)} labels to {output}")


if __name__ == "__main__":
    main()
