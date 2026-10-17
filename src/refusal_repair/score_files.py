"""Score-file provenance shared by scoring, aggregation, and comparisons."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path

from refusal_repair.datasets import file_sha256, read_jsonl, write_jsonl
from refusal_repair.evaluation import LlamaGuardJudge, label_response


@lru_cache(maxsize=64)
def _cached_sha256(path: Path, size: int, mtime_ns: int) -> str:
    # File size and modification time invalidate cached hashes between score files.
    return file_sha256(path)


def _guard_fingerprint(root: Path) -> dict:
    config = json.loads((root / "configs/experiment.json").read_text())
    model_dir = (root / config["judge_path"]).resolve()
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No Llama Guard weights found: {model_dir}")
    metadata = [model_dir / name for name in (
        "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
        "tokenizer.model", "special_tokens_map.json", "model.safetensors.index.json",
    ) if (model_dir / name).is_file()]
    files = {}
    for path in weights + metadata:
        stat = path.stat()
        files[path.name] = _cached_sha256(path, stat.st_size, stat.st_mtime_ns)
    return {"model_path": config["judge_path"], "files": files}


def _fingerprint(root: Path, source: Path, guard: bool) -> dict:
    package = Path(__file__).resolve().parent
    return {
        "input_sha256": file_sha256(source),
        "scorer_sha256": {name: file_sha256(package / name) for name in (
            "evaluation.py", "refusal.py", "score_files.py",
        )},
        "guard": _guard_fingerprint(root) if guard else None,
    }


def load_scored(root: Path, path: Path, guard: bool | None = None) -> list[dict]:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"Missing score provenance or unfinished scores; recompute to a new output: {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported score provenance; recompute: {path}")
    used_guard = manifest["fingerprint"]["guard"] is not None
    if guard is not None and guard != used_guard:
        raise ValueError(f"Cached scores use a different Guard setting: {path}")
    source = path.parent / manifest["input_path"]
    if manifest["fingerprint"] != _fingerprint(root, source, used_guard):
        raise ValueError(f"Stale score provenance; input, scorer, or Guard model changed: {path}")
    if file_sha256(path) != manifest["output_sha256"]:
        raise ValueError(f"Scored output changed after evaluation: {path}")
    rows = read_jsonl(path)
    if len(rows) != manifest["rows"]:
        raise ValueError(f"Scored row count does not match provenance: {path}")
    return rows


def score_file(root: Path, source: Path, output: Path, guard: bool = False, judge=None) -> list[dict]:
    if output.exists() or output.with_suffix(".manifest.json").exists():
        rows = load_scored(root, output, guard=guard)
        manifest = json.loads(output.with_suffix(".manifest.json").read_text())
        if (output.parent / manifest["input_path"]).resolve() != source.resolve():
            raise ValueError(f"Cached scores belong to a different input file: {output}")
        return rows
    fingerprint = _fingerprint(root, source, guard)
    rows = read_jsonl(source)
    if not rows:
        raise ValueError(f"Cannot score an empty response file: {source}")
    if guard and judge is None:
        config = json.loads((root / "configs/experiment.json").read_text())
        judge = LlamaGuardJudge(root / config["judge_path"])
    scored = [
        label_response(row, judge.unsafe_answer(row["prompt"], row["response"]) if guard else None)
        for row in rows
    ]
    if fingerprint != _fingerprint(root, source, guard):
        raise ValueError("Scoring inputs changed during evaluation; no scores were saved")
    write_jsonl(output, scored)
    manifest = {
        "format_version": 1,
        "input_path": os.path.relpath(source.resolve(), output.parent.resolve()),
        "fingerprint": fingerprint,
        "rows": len(scored),
        "output_sha256": file_sha256(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return scored
