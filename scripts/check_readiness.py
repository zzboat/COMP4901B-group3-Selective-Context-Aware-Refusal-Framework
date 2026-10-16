from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check baseline assets and runtime without loading a model")
    parser.add_argument("--hash-models", action="store_true", help="read all model weights to verify SHA-256")
    args = parser.parse_args()
    errors: list[str] = []

    if sys.version_info[:2] != (3, 9):
        errors.append(f"Python 3.9 required; found {sys.version.split()[0]}")
    expected = {"torch": "2.4.1", "transformers": "4.46.3", "accelerate": None,
                "numpy": None, "scikit-learn": None, "safetensors": None}
    for package, version in expected.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {package}")
            continue
        if version and actual.split("+")[0] != version:
            errors.append(f"{package} version should be {version}; found {actual}")

    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("PyTorch cannot access CUDA")
        else:
            device = torch.cuda.get_device_properties(0)
            memory_gib = device.total_memory / 1024**3
            print(f"GPU: {device.name}, {memory_gib:.1f} GiB; CUDA runtime {torch.version.cuda}")
            if memory_gib < 20:
                errors.append(f"Expected a GPU with at least 20 GiB; found {memory_gib:.1f} GiB")
    except Exception as exc:
        errors.append(f"CUDA probe failed: {exc}")

    model_manifest = json.loads((ROOT / "configs/model_manifest.json").read_text())
    for model in model_manifest["models"]:
        model_dir = ROOT / model["local_directory"]
        for filename, spec in model["weights"].items():
            path = model_dir / filename
            if not path.is_file():
                errors.append(f"Missing model weight: {path}")
                continue
            if path.stat().st_size != spec["bytes"]:
                errors.append(f"Wrong file size: {path}")
            if args.hash_models and sha256(path) != spec["sha256"]:
                errors.append(f"SHA-256 mismatch: {path}")
        for filename in ("config.json", "tokenizer_config.json"):
            if not (model_dir / filename).is_file():
                errors.append(f"Missing model/tokenizer file: {model_dir / filename}")

    data_manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for dataset in data_manifest["datasets"]:
        path = ROOT / dataset["path"]
        if not path.is_file():
            errors.append(f"Missing raw dataset: {path}")
        elif sha256(path) != dataset["sha256"]:
            errors.append(f"Raw dataset SHA-256 mismatch: {path}")

    split_manifest_path = ROOT / "data/splits/manifest.json"
    if not split_manifest_path.is_file():
        errors.append("Run scripts/prepare_splits.py to create the fixed data splits")
    else:
        split_manifest = json.loads(split_manifest_path.read_text())
        for name, spec in split_manifest.get("splits", {}).items():
            path = ROOT / "data/splits" / f"{name}.jsonl"
            if not path.is_file() or sha256(path) != spec["sha256"]:
                errors.append(f"Missing or changed split: {path}")
                continue
            with path.open(encoding="utf-8") as stream:
                count = sum(1 for line in stream if line.strip())
            if count != spec["rows"]:
                errors.append(f"Wrong row count for {path}: {count}")
        expected_sources = {dataset["name"]: dataset["sha256"] for dataset in data_manifest["datasets"]}
        if split_manifest.get("source_sha256") != expected_sources:
            errors.append("Split manifest source hashes do not match the raw data manifest")

    surgical = ROOT / "third_party/False-Refusal-Mitigation"
    if not (surgical / "UPSTREAM_COMMIT").is_file() or (surgical / "UPSTREAM_COMMIT").read_text().strip() != "916dcac6fb1c5fcccc9581471370645763c65ce6":
        errors.append("Surgical reference files are missing or not pinned to the recorded source revision")
    for name in ("harmful_train.json", "harmless_train.json", "harmful_val.json", "harmless_val.json"):
        if not (surgical / "dataset/splits" / name).is_file():
            errors.append(f"Missing Surgical reference file: {name}")

    adasteer = ROOT / "third_party/AdaSteer"
    if not (adasteer / "UPSTREAM_COMMIT").is_file() or (adasteer / "UPSTREAM_COMMIT").read_text().strip() != "2a02b78bea83df497dd3495e7a0dda4580f2dd53":
        errors.append("AdaSteer source package is missing or not pinned to the recorded source revision")
    vector_files = ("RD/mean_diff.pkl", "RD/class_a.pkl", "RD/class_b.pkl", "HD/proj.pkl", "HD/class_a.pkl", "HD/class_b.pkl")
    for name in vector_files:
        if not (adasteer / "vectors/llama31-8b-instruct" / name).is_file():
            errors.append(f"Missing AdaSteer vector: {name}")
    generator = adasteer / "adasteer/src/main_generate_steering_multi_adasteer.py"
    if generator.is_file() and 'attn_implementation="sdpa"' not in generator.read_text(encoding="utf-8"):
        errors.append("AdaSteer SDPA compatibility patch is missing")

    if errors:
        print("NOT READY")
        for error in errors:
            print(f"- {error}")
        return 1
    print("READY: baseline dependencies, GPU, model files, raw data, fixed splits, and author assets are present.")
    if not args.hash_models:
        print("Model weight SHA-256 checks were skipped; pass --hash-models to verify full weight contents.")
    print("No model weights were loaded and no inference was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
