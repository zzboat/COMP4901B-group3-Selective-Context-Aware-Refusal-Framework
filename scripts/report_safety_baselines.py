"""Build the two-benchmark safety report from verified, paired baseline outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from refusal_repair.datasets import file_sha256  # noqa: E402
from refusal_repair.evaluation import paired_result  # noqa: E402
from refusal_repair.score_files import load_scored  # noqa: E402

METHODS = ("original", "generic_rewrite", "retry_rewrite", "surgical", "adasteer")
SPLITS = {"xstest_unsafe_test": 200, "sorry_test": 220}


def validate_rows(rows, reference, method, expected_n):
    if len(rows) != expected_n or len({r["id"] for r in rows}) != expected_n:
        raise ValueError(f"Wrong row count or duplicate IDs for {method}")
    for row in rows:
        if row["method"] != method or row["label"] != "unsafe":
            raise ValueError(f"Unexpected method or reference label for {row['id']}")
        if any(type(row.get(key)) is not bool for key in ("refusal_heuristic", "guard_unsafe_answer")):
            raise ValueError(f"Missing boolean safety/refusal labels for {row['id']}")
    identity = lambda items: {r["id"]: (r["prompt"], r["dataset"], r["label"]) for r in items}
    if identity(rows) != identity(reference):
        raise ValueError(f"Prompt IDs, text or dataset differ for {method}")


def build_report(data, hashes, generation_commits):
    lines = [
        "# Baseline safety results: XSTest Unsafe and SORRY-Bench",
        "",
        "This report covers five methods on the same 200 XSTest Unsafe and 220 SORRY-Bench requests. "
        "Generation and scoring files are validated before aggregation; raw responses and model weights remain outside Git.",
        "",
        "## Evaluation protocol",
        "",
        "- Generator: local Llama 3.1 8B Instruct checkpoint; greedy decoding; 128-token generation cap.",
        "- Safety judge: Llama Guard 3 1B, with the original request and final answer in its conversation template.",
        "- **Guard unsafe-answer rate is a proxy, not the official SORRY-Bench harmful-compliance metric.**",
        "- Refusal rate uses the repository's English regex on the first 300 response characters. "
        "A non-refusal is neither proof of harmful compliance nor proof of task success.",
        "- Differences use identical prompt IDs/text. 95% paired percentile-bootstrap intervals use 2,000 resamples, seed 42. "
        "A non-significant difference is not evidence of safety equivalence.",
        "- `generic_rewrite` adds a fixed contextual instruction; `retry_rewrite` applies that same instruction only after a regex-detected initial refusal.",
        "- `surgical` denotes the archived baseline run, not the later S01 alpha-0.25 configuration. "
        "Its recorded generation revision implements full projection removal; the raw manifest does not record historical uncommitted changes.",
        "- AdaSteer uses the pinned author implementation and published vectors described in `docs/BASELINES.md`. "
        "Its short pre-pass and full generation count as two calls, but it has no comparable per-request timing/token log.",
        "",
        "## Results",
        "",
        "Lower Guard unsafe-answer rate is better. Refusal rate is reported separately and is not used as a substitute safety metric. "
        "Deltas are method minus original in percentage points (pp).",
        "",
        "| Benchmark | Method | N | Guard unsafe (count / rate) | Refusal (count / rate) | Guard delta, pp [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    transitions = []
    for split in SPLITS:
        original = data[(split, "original")]
        base = {r["id"]: r for r in original}
        for method in METHODS:
            rows = data[(split, method)]
            n = len(rows)
            unsafe = sum(r["guard_unsafe_answer"] for r in rows)
            refused = sum(r["refusal_heuristic"] for r in rows)
            delta = "reference"
            if method != "original":
                result = paired_result(original, rows, "guard_unsafe_answer")
                delta = f"{result['delta'] * 100:+.2f} [{result['ci_low'] * 100:+.2f}, {result['ci_high'] * 100:+.2f}]"
                added = sum(not base[r["id"]]["guard_unsafe_answer"] and r["guard_unsafe_answer"] for r in rows)
                removed = sum(base[r["id"]]["guard_unsafe_answer"] and not r["guard_unsafe_answer"] for r in rows)
                transitions.append(f"| {split} | {method} | {added} | {removed} | {added - removed:+d} |")
            lines.append(f"| {split} | {method} | {n} | {unsafe} / {unsafe / n:.2%} | {refused} / {refused / n:.2%} | {delta} |")
    lines += [
        "", "## Paired safety-label changes", "",
        "New unsafe labels are original-safe → method-unsafe; removed unsafe labels are the reverse. "
        "Report both because the net difference can hide newly introduced failures. These remain judge labels pending independent review.",
        "", "| Benchmark | Method | New unsafe labels | Removed unsafe labels | Net count |",
        "| --- | --- | ---: | ---: | ---: |", *transitions,
        "", "## Reproduction", "", "```bash",
        "for method in original generic_rewrite retry_rewrite surgical adasteer; do",
        "  for split in xstest_unsafe_test sorry_test; do",
        '    python scripts/score_results.py --input "results/raw/$method/$split.jsonl" --guard',
        "  done", "done", "python scripts/report_safety_baselines.py", "```", "",
        "Existing matching scores are verified and reused. Stale or incomplete scores are rejected rather than overwritten. "
        "The report command also refuses to replace a different existing report; use `--output` for a new destination.",
        "", "## Provenance", "",
        "Generation revisions recorded by these files: " + ", ".join(f"`{commit}`" for commit in sorted(generation_commits)) + ".",
        "",
        "All ten files share the same generator/config fingerprints and safety-judge/scorer fingerprints; "
        "each split also shares its generation split fingerprint. Scored manifests record the exact input, scorer, judge and output hashes.",
        "", "| Scored file (under `results/raw/`) | SHA-256 |", "| --- | --- |",
    ]
    lines += [f"| `{name}` | `{value}` |" for name, value in sorted(hashes.items())]
    lines += ["", "## Limits", "",
        "These results describe one model/checkpoint and one fixed prompt selection. "
        "The 128-token cap can truncate answers and limits both safety and completeness interpretation. "
        "No independent human or official SORRY-Bench adjudication is implied. "
        "This report does not pool the two benchmarks or include later exploratory intervention experiments.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/BASELINE_SAFETY_RESULTS.md")
    args = parser.parse_args()
    data, hashes, commits = {}, {}, set()
    common_generation = (file_sha256(ROOT / "configs/model_manifest.json"),
                         file_sha256(ROOT / "configs/experiment.json"))
    common_scoring = None
    for split, expected_n in SPLITS.items():
        split_hash = None
        for method in METHODS:
            source = ROOT / "results/raw" / method / f"{split}.jsonl"
            path = source.with_suffix(".scored.jsonl")
            rows = load_scored(ROOT, path, guard=True)
            generation = json.loads(source.with_suffix(".manifest.json").read_text())
            scoring = json.loads(path.with_suffix(".manifest.json").read_text())
            if generation["output_sha256"] != file_sha256(source) or generation["rows"] != len(rows):
                raise ValueError(f"Generation provenance differs: {source}")
            if (generation["method"] != method or generation["split"] != split
                    or scoring["fingerprint"]["input_sha256"] != generation["output_sha256"]):
                raise ValueError(f"Scores or manifest belong to a different generation: {source}")
            gen_identity = (generation["model_manifest_sha256"], generation["config_sha256"])
            score_identity = (scoring["fingerprint"]["guard"], scoring["fingerprint"]["scorer_sha256"])
            if common_scoring is None:
                common_scoring = score_identity
            if gen_identity != common_generation or score_identity != common_scoring:
                raise ValueError("Methods do not share the same model/config/judge/scorer fingerprints")
            if split_hash is None:
                split_hash = generation["split_sha256"]
            if split_hash != generation["split_sha256"]:
                raise ValueError(f"Generation split fingerprint differs: {source}")
            reference = rows if method == "original" else data[(split, "original")]
            validate_rows(rows, reference, method, expected_n)
            data[(split, method)] = rows
            hashes[f"{method}/{path.name}"] = file_sha256(path)
            commits.add(generation["code_commit"])
    report = build_report(data, hashes, commits)
    if args.output.exists() and args.output.read_text(encoding="utf-8") != report:
        raise FileExistsError(f"Existing report preserved; choose a new --output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        args.output.write_text(report, encoding="utf-8")
    print(f"Saved or verified {args.output}")


if __name__ == "__main__":
    main()
