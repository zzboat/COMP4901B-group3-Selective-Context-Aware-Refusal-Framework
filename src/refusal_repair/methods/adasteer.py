"""Run the pinned AdaSteer author implementation without vendoring its code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

UPSTREAM_COMMIT = "2a02b78bea83df497dd3495e7a0dda4580f2dd53"


def run_adasteer(root: Path, upstream: Path, rows: list[dict], split: str, model_path: Path) -> list[dict]:
    upstream = upstream.resolve()
    if not (upstream / "adasteer/src/main_generate_steering_multi_adasteer.py").is_file():
        raise FileNotFoundError(f"AdaSteer author checkout missing: {upstream}")
    revision_file = upstream / "UPSTREAM_COMMIT"
    if not revision_file.is_file():
        raise FileNotFoundError(f"AdaSteer source revision marker missing: {revision_file}")
    commit = revision_file.read_text(encoding="utf-8").strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"AdaSteer checkout must be pinned to {UPSTREAM_COMMIT}; found {commit}")
    generator = upstream / "adasteer/src/main_generate_steering_multi_adasteer.py"
    if 'attn_implementation="sdpa"' not in generator.read_text(encoding="utf-8"):
        raise ValueError("AdaSteer must use the documented SDPA compatibility patch")
    vector = upstream / "vectors/llama31-8b-instruct/RD/mean_diff.pkl"
    if not vector.is_file():
        raise FileNotFoundError(vector)

    dataset_name = f"baseline_{split}"
    input_dir = root / "results/raw/adasteer_input"
    output_dir = root / "results/raw/adasteer_author"
    input_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generate").mkdir(parents=True, exist_ok=True)
    (input_dir / f"{dataset_name}.json").write_text(
        json.dumps([{"input": row["prompt"]} for row in rows], ensure_ascii=False), encoding="utf-8"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(upstream) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(upstream / "adasteer/src/main_generate_steering_multi_adasteer.py"),
        "--model_name_or_path", str(model_path.resolve()),
        "--data_dir", str(input_dir.resolve()),
        "--output_dir", str(output_dir.resolve()),
        "--model_sign", "llama31",
        "--dataset_list", dataset_name,
        "--steer_vector", str(vector),
        "--alpha", "0",
        "--overwrite", "True",
        "--if_all_layers", "True",
        "--if_support_sys_prompt", "False",
        "--bs", "1",
    ]
    subprocess.run(command, cwd=upstream, env=env, check=True)
    author_output = json.loads((output_dir / "generate" / f"{dataset_name}.json").read_text(encoding="utf-8"))
    if len(author_output) != len(rows):
        raise ValueError("AdaSteer returned a different number of responses")
    results = []
    for row, answer in zip(rows, author_output):
        if answer.get("input") != row["prompt"]:
            raise ValueError("AdaSteer output order differs from the input order")
        results.append({**row, "method": "adasteer", "response": answer["output"], "model_calls": 2})
    return results
