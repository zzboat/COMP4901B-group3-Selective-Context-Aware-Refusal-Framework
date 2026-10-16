from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refusal_repair.datasets import prepare_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed, checksum-verified benchmark splits")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_splits(args.root, args.seed)["splits"], indent=2))


if __name__ == "__main__":
    main()
