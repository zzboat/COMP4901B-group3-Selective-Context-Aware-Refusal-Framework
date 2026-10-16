# Research assets

## Public benchmark data on the server

| Dataset | Local file | Published rows | Project plan | Source |
| --- | --- | ---: | ---: | --- |
| XSTest | `data/raw/xstest_prompts.csv` | 450 | 250 safe + 200 unsafe | https://github.com/paul-rottger/xstest |
| OR-Bench Hard-1K | `data/raw/or-bench-hard-1k.csv` | 1,319 | 1,000 | https://huggingface.co/datasets/bench-llm/or-bench |
| OR-Bench Toxic | `data/raw/or-bench-toxic.csv` | 655 | 600 | https://huggingface.co/datasets/bench-llm/or-bench |

The full CSVs are downloaded. Their exact SHA-256 hashes are recorded in `data/manifest.json`. The two public OR-Bench files were retrieved through hf-mirror.com after direct Hugging Face access timed out. For proposal-matched results, define and record fixed 1,000/600-row subsets; do not silently mix full-file and subset scores.

## Model weights

| Role | Model | Source | Local directory |
| --- | --- | --- | --- |
| Main generator | Meta Llama 3.1 8B Instruct | https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct | `models/Meta-Llama-3.1-8B-Instruct/` |
| Prompt-attack detector | Llama Prompt Guard 2 86M | https://modelscope.cn/models/LLM-Research/Llama-Prompt-Guard-2-86M | `models/Llama-Prompt-Guard-2-86M/` |
| Content-safety verifier | Llama Guard 3 1B | https://modelscope.cn/models/LLM-Research/Llama-Guard-3-1B | `models/Llama-Guard-3-1B/` |

Model files are held in `/root/shared-nvme/COMP4091B-models/`, which `models/` links to. Download the root Safetensors and tokenizer/config files for Llama 3.1; exclude its `original/` folder to avoid a duplicate weight format. These weights are ignored by Git.

Prompt Guard 2 86M detects prompt injection and jailbreak attempts. Its labels do not directly indicate whether an ordinary user request is harmful. Llama Guard 3 1B is the selected content-safety verifier. Its error rate on the project benchmarks still needs to be measured before using its output as a gate.

## Restricted harm-side data

The optional SORRY-Bench 2025/03 dataset is at https://huggingface.co/datasets/sorry-bench/sorry-bench-202503 . Its base `question.jsonl` has 440 prompts and is now stored at `data/raw/sorry-bench-202503/question.jsonl` after approved access. The plan calls for a documented 220-prompt subset. The dataset license prohibits third-party transfers. Keep raw prompts out of GitHub and do not distribute copies to teammates who lack their own access.

The server currently cannot reach huggingface.co directly. Use a trusted network route for authenticated download, or obtain explicit consent before routing an account token through a third-party mirror. Keep all tokens out of the repository and command history.
