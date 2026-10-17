"""Load local benchmark files and create immutable, disjoint experiment splits."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, list) and all(isinstance(row, dict) for row in records):
            return records
    raise ValueError(f"Expected a JSON array or JSONL file of objects: {path}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path, dataset: str, label: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "prompt" not in rows[0]:
        raise ValueError(f"Missing prompt column: {path}")
    return [
        {
            "id": f"{dataset}:{row.get('id') or index}",
            "dataset": dataset,
            "label": row.get("label", label),
            "category": row.get("category") or row.get("type", ""),
            "prompt": row["prompt"].strip(),
        }
        for index, row in enumerate(rows, 1)
    ]


def _sorry_rows(path: Path) -> list[dict]:
    result = []
    for row in read_jsonl(path):
        turns = row.get("turns")
        if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], str):
            raise ValueError("SORRY-Bench base data must contain one text turn per question")
        result.append(
            {
                "id": f"sorry:{row['question_id']}",
                "dataset": "sorry",
                "label": "unsafe",
                "category": str(row["category"]),
                "prompt": turns[0].strip(),
            }
        )
    return result


def _sample(rows: list[dict], size: int, rng: random.Random) -> list[dict]:
    if len(rows) < size:
        raise ValueError(f"Need {size} rows; found {len(rows)}")
    return rng.sample(rows, size)


def prepare_splits(root: Path, seed: int = 42) -> dict:
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["datasets"]:
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Dataset checksum changed: {path}")

    rng = random.Random(seed)
    xstest = _csv_rows(root / "data/raw/xstest_prompts.csv", "xstest", "")
    hard = _csv_rows(root / "data/raw/or-bench-hard-1k.csv", "or_hard", "safe")
    toxic = _csv_rows(root / "data/raw/or-bench-toxic.csv", "or_toxic", "unsafe")
    sorry = _sorry_rows(root / "data/raw/sorry-bench-202503/question.jsonl")

    if Counter(row["label"] for row in xstest) != {"safe": 250, "unsafe": 200}:
        raise ValueError("Unexpected XSTest labels or row counts")
    if len(sorry) != 440 or set(Counter(row["category"] for row in sorry).values()) != {10}:
        raise ValueError("Expected the SORRY-Bench 2025/03 base dataset")
    if any(not row["prompt"] for row in xstest + hard + toxic + sorry):
        raise ValueError("A benchmark row has an empty prompt")

    hard_1000 = _sample(hard, 1000, rng)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in sorry:
        by_category[row["category"]].append(row)
    sorry_220 = [row for category in sorted(by_category) for row in _sample(by_category[category], 5, rng)]

    splits = {
        "or_hard_extract": hard_1000[:300],
        "or_hard_valid": hard_1000[300:500],
        "or_hard_test": hard_1000[500:],
        "or_toxic_test": _sample(toxic, 600, rng),
        "xstest_safe_test": [row for row in xstest if row["label"] == "safe"],
        "xstest_unsafe_test": [row for row in xstest if row["label"] == "unsafe"],
        "sorry_test": sorry_220,
    }
    all_ids = [row["id"] for rows in splits.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A prompt appears in more than one split")
    normalized_prompts = [" ".join(row["prompt"].casefold().split()) for rows in splits.values() for row in rows]
    if len(normalized_prompts) != len(set(normalized_prompts)):
        raise ValueError("A prompt text appears in more than one split")

    output_dir = root / "data/splits"
    output_manifest = {
        "seed": seed,
        "source_sha256": {item["name"]: item["sha256"] for item in manifest["datasets"]},
        "splits": {},
    }
    with tempfile.TemporaryDirectory(prefix=".splits-", dir=output_dir.parent) as directory:
        staged = Path(directory)
        for name, rows in splits.items():
            path = staged / f"{name}.jsonl"
            write_jsonl(path, rows)
            output_manifest["splits"][name] = {"rows": len(rows), "sha256": file_sha256(path)}
        (staged / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists() and any(output_dir.iterdir()):
            existing = output_dir / "manifest.json"
            if not existing.is_file() or json.loads(existing.read_text()) != output_manifest:
                raise FileExistsError("Fixed splits already exist with different settings; use a separate data root")
            for name, spec in output_manifest["splits"].items():
                path = output_dir / f"{name}.jsonl"
                if not path.is_file() or file_sha256(path) != spec["sha256"]:
                    raise ValueError(f"Existing fixed split is missing or changed; preserved: {path}")
            return output_manifest
        staged.rename(output_dir)
    return output_manifest


def load_split(root: Path, name: str) -> list[dict]:
    manifest_path = root / "data/splits/manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run scripts/prepare_splits.py before inference")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if name not in manifest["splits"]:
        raise KeyError(f"Unknown split: {name}")
    path = root / "data/splits" / f"{name}.jsonl"
    if file_sha256(path) != manifest["splits"][name]["sha256"]:
        raise ValueError(f"Split changed after preparation: {path}")
    return read_jsonl(path)
