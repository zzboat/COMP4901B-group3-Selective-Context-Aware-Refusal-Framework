from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refusal_repair.datasets import file_sha256, load_split, read_records  # noqa: E402
from refusal_repair.inference import HFEngine  # noqa: E402
from refusal_repair.methods.surgical import extract_vector, save_vector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Surgical on disjoint development data")
    reference = ROOT / "third_party/False-Refusal-Mitigation/dataset/splits"
    parser.add_argument("--harmful-train", type=Path, default=reference / "harmful_train.json")
    parser.add_argument("--harmless-train", type=Path, default=reference / "harmless_train.json")
    parser.add_argument("--harmful-valid", type=Path, default=reference / "harmful_val.json")
    parser.add_argument("--harmless-valid", type=Path, default=reference / "harmless_val.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/surgical_vector.pt")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    engine = HFEngine(ROOT / config["model_path"], config["dtype"], config["max_new_tokens"])
    artifact = extract_vector(
        engine,
        read_records(args.harmful_train),
        read_records(args.harmless_train),
        load_split(ROOT, "or_hard_extract"),
        read_records(args.harmful_valid),
        read_records(args.harmless_valid),
        load_split(ROOT, "or_hard_valid"),
        config["surgical"],
        config["seed"],
    )
    artifact["model_manifest_sha256"] = file_sha256(ROOT / "configs/model_manifest.json")
    artifact["source_sha256"] = {
        "harmful_train": file_sha256(args.harmful_train),
        "harmless_train": file_sha256(args.harmless_train),
        "harmful_valid": file_sha256(args.harmful_valid),
        "harmless_valid": file_sha256(args.harmless_valid),
        "pseudo_train": file_sha256(ROOT / "data/splits/or_hard_extract.jsonl"),
        "pseudo_valid": file_sha256(ROOT / "data/splits/or_hard_valid.jsonl"),
    }
    save_vector(args.output, artifact)
    print(json.dumps({key: value for key, value in artifact.items() if key != "unit"}, indent=2))


if __name__ == "__main__":
    main()
