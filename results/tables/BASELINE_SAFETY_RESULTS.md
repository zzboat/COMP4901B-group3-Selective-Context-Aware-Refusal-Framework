# Baseline safety results: XSTest Unsafe and SORRY-Bench

This report covers five methods on the same 200 XSTest Unsafe and 220 SORRY-Bench requests. Generation and scoring files are validated before aggregation; raw responses and model weights remain outside Git.

## Evaluation protocol

- Generator: local Llama 3.1 8B Instruct checkpoint; greedy decoding; 128-token generation cap.
- Safety judge: Llama Guard 3 1B, with the original request and final answer in its conversation template.
- **Guard unsafe-answer rate is a proxy, not the official SORRY-Bench harmful-compliance metric.**
- Refusal rate uses the repository's English regex on the first 300 response characters. A non-refusal is neither proof of harmful compliance nor proof of task success.
- Differences use identical prompt IDs/text. 95% paired percentile-bootstrap intervals use 2,000 resamples, seed 42. A non-significant difference is not evidence of safety equivalence.
- `generic_rewrite` adds a fixed contextual instruction; `retry_rewrite` applies that same instruction only after a regex-detected initial refusal.
- `surgical` denotes the archived baseline run, not the later S01 alpha-0.25 configuration. Its recorded generation revision implements full projection removal; the raw manifest does not record historical uncommitted changes.
- AdaSteer uses the pinned author implementation and published vectors described in `docs/BASELINES.md`. Its short pre-pass and full generation count as two calls, but it has no comparable per-request timing/token log.

## Results

Lower Guard unsafe-answer rate is better. Refusal rate is reported separately and is not used as a substitute safety metric. Deltas are method minus original in percentage points (pp).

| Benchmark | Method | N | Guard unsafe (count / rate) | Refusal (count / rate) | Guard delta, pp [95% CI] |
| --- | --- | ---: | ---: | ---: | --- |
| xstest_unsafe_test | original | 200 | 3 / 1.50% | 166 / 83.00% | reference |
| xstest_unsafe_test | generic_rewrite | 200 | 1 / 0.50% | 192 / 96.00% | -1.00 [-3.00, +0.50] |
| xstest_unsafe_test | retry_rewrite | 200 | 3 / 1.50% | 161 / 80.50% | +0.00 [+0.00, +0.00] |
| xstest_unsafe_test | surgical | 200 | 12 / 6.00% | 142 / 71.00% | +4.50 [+1.00, +8.00] |
| xstest_unsafe_test | adasteer | 200 | 22 / 11.00% | 99 / 49.50% | +9.50 [+5.00, +14.00] |
| sorry_test | original | 220 | 28 / 12.73% | 167 / 75.91% | reference |
| sorry_test | generic_rewrite | 220 | 11 / 5.00% | 191 / 86.82% | -7.73 [-12.27, -3.18] |
| sorry_test | retry_rewrite | 220 | 32 / 14.55% | 161 / 73.18% | +1.82 [+0.45, +3.64] |
| sorry_test | surgical | 220 | 53 / 24.09% | 133 / 60.45% | +11.36 [+6.82, +16.36] |
| sorry_test | adasteer | 220 | 53 / 24.09% | 134 / 60.91% | +11.36 [+5.45, +17.27] |

## Paired safety-label changes

New unsafe labels are original-safe → method-unsafe; removed unsafe labels are the reverse. Report both because the net difference can hide newly introduced failures. These remain judge labels pending independent review.

| Benchmark | Method | New unsafe labels | Removed unsafe labels | Net count |
| --- | --- | ---: | ---: | ---: |
| xstest_unsafe_test | generic_rewrite | 1 | 3 | -2 |
| xstest_unsafe_test | retry_rewrite | 0 | 0 | +0 |
| xstest_unsafe_test | surgical | 11 | 2 | +9 |
| xstest_unsafe_test | adasteer | 21 | 2 | +19 |
| sorry_test | generic_rewrite | 6 | 23 | -17 |
| sorry_test | retry_rewrite | 4 | 0 | +4 |
| sorry_test | surgical | 28 | 3 | +25 |
| sorry_test | adasteer | 37 | 12 | +25 |

## Reproduction

```bash
for method in original generic_rewrite retry_rewrite surgical adasteer; do
  for split in xstest_unsafe_test sorry_test; do
    python scripts/score_results.py --input "results/raw/$method/$split.jsonl" --guard
  done
done
python scripts/report_safety_baselines.py
```

Existing matching scores are verified and reused. Stale or incomplete scores are rejected rather than overwritten. The report command also refuses to replace a different existing report; use `--output` for a new destination.

## Provenance

Generation revisions recorded by these files: `986714ffc6770f0f70999c4e847d424b107846d3`.

All ten files share the same generator/config fingerprints and safety-judge/scorer fingerprints; each split also shares its generation split fingerprint. Scored manifests record the exact input, scorer, judge and output hashes.

| Scored file (under `results/raw/`) | SHA-256 |
| --- | --- |
| `adasteer/sorry_test.scored.jsonl` | `050658180c0b64fc7bcc0ff0e6486e57c6a7429a7ddb285f00158e7681eb3118` |
| `adasteer/xstest_unsafe_test.scored.jsonl` | `f0f17e96f69dc2592ae7dd284b35f98b00bca83aa957e0578cc25fad14c57c54` |
| `generic_rewrite/sorry_test.scored.jsonl` | `de56c4075d173d9a05fe95953575cac62daac83399ff7d81000f2c0012c8cc30` |
| `generic_rewrite/xstest_unsafe_test.scored.jsonl` | `49b45990989e771aba5f15698aaf3c5ca1e0f49dea554c81e53d00f94d6adba2` |
| `original/sorry_test.scored.jsonl` | `9af21687130e43b0a11f83bbbf0e28d6c6be4c66c8e7889eb21ee24de75d4b21` |
| `original/xstest_unsafe_test.scored.jsonl` | `70bbf38f0a929ddfbbd24307b4436d4d6e97100701f3335e9b4e277c485751f3` |
| `retry_rewrite/sorry_test.scored.jsonl` | `94548c5602659d57ff1b81717b089c7d565a60eb7d94c5c02d97e3eb25cce3e7` |
| `retry_rewrite/xstest_unsafe_test.scored.jsonl` | `edf1ec89d028b668f5fa04e702674b208b0ed76c59dec3df8d3c9a012cfab500` |
| `surgical/sorry_test.scored.jsonl` | `1b90a794d20bea9f34fc2c9c1a1f845805b4b97481037e2a93709815038f8fd2` |
| `surgical/xstest_unsafe_test.scored.jsonl` | `813edb2eaa3c919c44926c98d63ed8849e6c28407af0671693a0289cf4eec827` |

## Limits

These results describe one model/checkpoint and one fixed prompt selection. The 128-token cap can truncate answers and limits both safety and completeness interpretation. No independent human or official SORRY-Bench adjudication is implied. This report does not pool the two benchmarks or include later exploratory intervention experiments.
