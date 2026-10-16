# Baseline reproduction (code only until an experiment is authorized)

This repository supports five comparison rows on the same local Llama 3.1 8B Instruct checkpoint: `original`, `generic_rewrite`, `retry_rewrite`, `surgical`, and `adasteer`. The code does not run a benchmark on import or installation. The commands below are for a later experimental run.

## Data protocol

`scripts/prepare_splits.py` verifies every source SHA-256 in `data/manifest.json` before writing ignored `data/splits/` files. Seed 42 selects 1,000 of the 1,319 OR-Bench Hard rows, divided into 300 extraction, 200 validation, and 500 held-out test rows. It selects 600 OR-Bench Toxic rows, retains all 250 safe and 200 unsafe XSTest rows as tests, and selects five SORRY-Bench 2025/03 prompts from each of its 44 categories (220 held-out prompts). It checks duplicate IDs and normalized prompt text across the generated splits. Do not tune on any `*_test` file.

The final test files contain the raw prompts and remain outside Git. The final SORRY-Bench test is used only for safety evaluation; it cannot measure false refusal because its base prompts are unsafe. The OR-Bench Hard test and XSTest safe test measure false refusal. XSTest unsafe, OR-Bench Toxic, and SORRY-Bench test measure safety. The same split files, chat template, greedy decoding, 128-token cap, and judge must be used for every method.

```bash
python scripts/prepare_splits.py
```

## Dependency and author-code setup

Use the isolated server environment; do not change system Python. Setup and readiness commands:

```bash
source /base/mambaforge/etc/profile.d/conda.sh
conda activate /root/shared-nvme/COMP4091B-baselines-py39
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-py39-cu121.lock --index-url https://pypi.tuna.tsinghua.edu.cn/simple
bash scripts/fetch_author_assets.sh
python scripts/check_readiness.py
```

The AdaSteer authors specify Python 3.9, PyTorch 2.4.1, and Transformers 4.46.3. The prepared CUDA build is 12.1. The RTX 3090 has 24 GiB; use batch size 1. FlashAttention is not required because the AdaSteer loader has the documented SDPA compatibility patch. This one environment serves every baseline: original and rewrite controls use the shared inference stack, Surgical also uses scikit-learn for vector selection, and AdaSteer uses the pinned author code and vectors.

`third_party/` is ignored by Git. The Surgical files are limited to the four harmful/harmless reference splits from pinned commit `916dcac6fb1c5fcccc9581471370645763c65ce6`. AdaSteer's runtime package and only the vectors loaded by its Llama 3.1 implementation are staged under ignored `third_party/AdaSteer/`; `UPSTREAM_COMMIT` records pinned source revision `2a02b78bea83df497dd3495e7a0dda4580f2dd53`.

AdaSteer hard-codes FlashAttention 2, which is optional and can be difficult to build on a shared RTX 3090 host. The prepared checkout applies a one-line compatibility patch to select PyTorch SDPA (`attn_implementation="sdpa"`) instead. This changes the attention kernel, not AdaSteer's steering method; record it in every result manifest. Check semantic near-duplicates between reference data and held-out tests before making a generalization claim.

## Surgical: adapted method reproduction

`scripts/extract_vectors.py` uses the Surgical authors' harmful and harmless train/validation data and this project's OR-Bench Hard extraction/validation splits. It searches residual-stream directions at the final three prompt tokens across decoder layers, subtracts the true-refusal projection from candidate false-refusal directions, filters candidates by first-token KL, and selects the candidate that most reduces pseudo-harmful refusal score on validation. The final unit vector is saved in ignored `artifacts/surgical_vector.pt`. Inference ablates that direction at all layers and token positions through removable hooks; original weights are never edited.

```bash
python scripts/extract_vectors.py
python scripts/run_benchmark.py --method surgical --split or_hard_test
```

This is a **Llama 3.1 adaptation** of the ICLR 2025 method. The published Surgical table uses Llama 3 8B, different datasets and a different judge; those numbers are context, not a directly comparable baseline. Our implementation uses the final three prompt token positions and a fixed refusal-token score. Document those choices if comparing against an exact author-code run.

## AdaSteer: pinned author implementation

`run_benchmark.py --method adasteer` invokes the pinned author code, using its published Llama 3.1 vectors and adaptive coefficients. It presents exactly the same local test prompts to the author implementation at batch size 1, then checks output count and order before converting responses to our JSONL format. The author implementation makes a short pre-pass before the full generation. It has a hard-coded 128-token limit and does not expose per-request timing; `mean_elapsed_sec` is therefore omitted for this row.

```bash
python scripts/run_benchmark.py --method adasteer --split or_hard_test
```

AdaSteer was published primarily as a jailbreak defense. Its paper reports XSTest but not SORRY-Bench. Author-supplied vectors are tied to their checkpoint and should be refitted if a materially different Llama 3.1 revision is used. Check the local model fingerprint in `configs/model_manifest.json` and report this limitation.

## Original and prompt controls

```bash
python scripts/run_benchmark.py --method original --split or_hard_test
python scripts/run_benchmark.py --method generic_rewrite --split or_hard_test
python scripts/run_benchmark.py --method retry_rewrite --split or_hard_test
```

Run each method separately on all five `*_test` splits. Existing output files are preserved rather than silently overwritten. Each run writes an ignored per-example JSONL file and a manifest with code, config, model-manifest, split, and output fingerprints.

## Evaluation

```bash
python scripts/score_results.py --input results/raw/original/or_hard_test.jsonl --guard
python scripts/summarize.py results/raw/*/*.scored.jsonl
python scripts/compare.py --original results/raw/original/or_hard_test.scored.jsonl --method results/raw/surgical/or_hard_test.scored.jsonl
```

The refusal regex is a screening label for over-refusal, not a measure of answer quality. `--guard` adds Llama Guard 3 1B's answer-safety label; `guard_unsafe_answer_rate` is a **proxy**, not the official SORRY-Bench harmful-compliance metric. For a paper-quality SORRY-Bench result, score the same saved answers with the [official SORRY-Bench judge](https://github.com/SORRY-Bench/sorry-bench) or a blinded human rubric and report the judge version. Manually review a stratified sample of benign answers for task completion and refusals. Do not use the same safety verifier as both a method gate and the only evaluation judge.

Source methods: [Surgical paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/53dbd7e34fab703a639964e2d3ee9e84-Paper-Conference.pdf), [Surgical code](https://github.com/mainlp/False-Refusal-Mitigation), [AdaSteer paper](https://aclanthology.org/2025.emnlp-main.1248.pdf), [AdaSteer code](https://github.com/MuyuenLP/AdaSteer).
