from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refusal_repair.datasets import file_sha256, load_split, write_jsonl  # noqa: E402
from refusal_repair.methods.adasteer import UPSTREAM_COMMIT, run_adasteer  # noqa: E402
from refusal_repair.methods.rewrite import run_rewrite  # noqa: E402
from refusal_repair.methods.surgical import load_vector  # noqa: E402

TEST_SPLITS = ("or_hard_test", "or_toxic_test", "xstest_safe_test", "xstest_unsafe_test", "sorry_test")
METHODS = ("original", "generic_rewrite", "retry_rewrite", "surgical", "adasteer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen baseline on one frozen test split")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--split", choices=TEST_SPLITS, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--surgical-vector", type=Path, default=ROOT / "artifacts/surgical_vector.pt")
    parser.add_argument("--adasteer-checkout", type=Path, default=ROOT / "third_party/AdaSteer")
    args = parser.parse_args()
    config_path = ROOT / "configs/experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = load_split(ROOT, args.split)
    output = args.output or ROOT / "results/raw" / args.method / f"{args.split}.jsonl"
    if output.exists():
        raise FileExistsError(f"Existing run preserved: {output}")
    model_path = ROOT / config["model_path"]
    if args.method == "adasteer":
        results = run_adasteer(ROOT, args.adasteer_checkout, rows, args.split, model_path)
    else:
        from refusal_repair.inference import HFEngine

        engine = HFEngine(model_path, config["dtype"], config["max_new_tokens"])
        unit = None
        if args.method == "surgical":
            artifact = load_vector(args.surgical_vector)
            if artifact.get("model_manifest_sha256") != file_sha256(ROOT / "configs/model_manifest.json"):
                raise ValueError("Surgical vector was fitted for a different model manifest")
            unit = artifact["unit"]
        results = []
        for row in rows:
            if args.method == "original":
                answer = {**engine.generate(row["prompt"]), "model_calls": 1}
            elif args.method == "surgical":
                answer = {**engine.generate(row["prompt"], unit), "model_calls": 1}
            else:
                answer = run_rewrite(engine, row["prompt"], conditional=args.method == "retry_rewrite")
            results.append({**row, "method": args.method, **answer})
    write_jsonl(output, results)
    manifest = {
        "method": args.method,
        "split": args.split,
        "rows": len(results),
        "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "config_sha256": file_sha256(config_path),
        "split_sha256": file_sha256(ROOT / "data/splits" / f"{args.split}.jsonl"),
        "model_manifest_sha256": file_sha256(ROOT / "configs/model_manifest.json"),
        "surgical_vector_sha256": file_sha256(args.surgical_vector) if args.method == "surgical" else None,
        "adasteer_upstream_commit": UPSTREAM_COMMIT if args.method == "adasteer" else None,
        "adasteer_generator_sha256": file_sha256(args.adasteer_checkout / "adasteer/src/main_generate_steering_multi_adasteer.py") if args.method == "adasteer" else None,
        "adasteer_vector_sha256s": ({
            name: file_sha256(args.adasteer_checkout / "vectors/llama31-8b-instruct" / name)
            for name in ("RD/mean_diff.pkl", "RD/class_a.pkl", "RD/class_b.pkl", "HD/proj.pkl", "HD/class_a.pkl", "HD/class_b.pkl")
        } if args.method == "adasteer" else None),
        "output_sha256": file_sha256(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} responses to {output}")


if __name__ == "__main__":
    main()
