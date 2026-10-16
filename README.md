# COMP4901B Group 3: Selective Context-Aware Refusal Repair

Training-free, uncertainty-aware refusal repair for safety-aligned LLMs. The working directory is named `COMP4091B` as requested; the course and GitHub repository use `COMP4901B`.

## Layout

- `configs/`: experiment parameters and model choices
- `data/raw/`: downloaded benchmark files (ignored by Git)
- `data/splits/`: deterministic evaluation subsets (ignored by Git)
- `data/manifest.json`: provenance, row counts, and SHA-256 checksums
- `models/`: link to weights on the shared NVMe volume (ignored by Git)
- `src/refusal_repair/`: framework code
- `scripts/`: reproducible setup and evaluation entry points
- `tests/`: tests for data processing and inference logic
- `results/raw/`: per-example outputs (ignored by Git)
- `results/tables/`, `results/figures/`: report-ready summaries
- `docs/`: setup and methods notes

## Assets

The public XSTest, OR-Bench Hard, and OR-Bench Toxic CSVs are already in `data/raw/`. Their sources and checksums are in `data/manifest.json`. The requested Llama 3.1 8B Instruct and Prompt Guard 2 86M weights are sourced from ModelScope. The optional SORRY-Bench requires the researcher's own Hugging Face access approval. See `docs/ASSETS.md`.

## Collaboration

Use GitHub pull requests for code, configurations, documentation, and small summary results. Keep weights, raw benchmark data, per-example model outputs, credentials, and local environments out of Git. Record the exact dataset selection and model revision used for each experiment.
