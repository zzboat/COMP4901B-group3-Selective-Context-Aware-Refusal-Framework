from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refusal_repair.score_files import score_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Label one generated-response file")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--guard", action="store_true", help="Also classify answer safety with Llama Guard 3 1B")
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".scored.jsonl")
    scored = score_file(ROOT, args.input, output, guard=args.guard)
    print(f"Saved or verified {len(scored)} labels at {output}")


if __name__ == "__main__":
    main()
