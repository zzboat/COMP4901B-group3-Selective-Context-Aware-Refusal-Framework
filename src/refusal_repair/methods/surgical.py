"""Llama 3.1 adaptation of Surgical's single-vector false-refusal ablation."""

from __future__ import annotations

import json
import random
from pathlib import Path


def orthogonalize(false_vector, true_vector, strength: float = 1.0):
    """Remove a chosen fraction of the false vector's true-refusal projection."""
    if not 0 <= strength <= 1:
        raise ValueError("Orthogonalization strength must be in [0, 1]")
    true_unit = true_vector / true_vector.norm().clamp_min(1e-8)
    result = false_vector - strength * (false_vector @ true_unit) * true_unit
    if result.norm().item() < 1e-8:
        raise ValueError("False-refusal direction vanishes after orthogonalization")
    return result / result.norm()


def _prompts(rows: list[dict]) -> list[str]:
    values = []
    for row in rows:
        prompt = row.get("prompt") or row.get("instruction")
        if prompt is None and isinstance(row.get("turns"), list) and len(row["turns"]) == 1:
            prompt = row["turns"][0]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Every reference row needs a non-empty prompt")
        values.append(prompt.strip())
    return values


def _select(engine, rows: list[dict], count: int, seed: int, require_refusal: bool) -> list[str]:
    prompts = _prompts(rows)
    random.Random(seed).shuffle(prompts)
    selected = []
    for prompt in prompts:
        if not require_refusal or engine.refusal_score(engine.first_logits(prompt)) > 0:
            selected.append(prompt)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(f"Need {count} qualifying prompts; found {len(selected)}")
    return selected


def _mean_hidden(engine, prompts: list[str], offsets: list[int]):
    total = None
    for prompt in prompts:
        hidden = engine.hidden_states(prompt, offsets)
        total = hidden if total is None else total + hidden
    return total / len(prompts)


def _score_drop(engine, prompts: list[str], base_scores: list[float], unit) -> float:
    changed = [engine.refusal_score(engine.first_logits(prompt, unit)) for prompt in prompts]
    return sum(before - after for before, after in zip(base_scores, changed)) / len(prompts)


def _first_token_kl(engine, prompts: list[str], base_logits: list, unit) -> float:
    torch = engine.torch
    values = []
    for prompt, original in zip(prompts, base_logits):
        edited = engine.first_logits(prompt, unit)
        log_p = torch.log_softmax(original, -1)
        log_q = torch.log_softmax(edited, -1)
        values.append((log_p.exp() * (log_p - log_q)).sum().item())
    return sum(values) / len(values)


def extract_vector(
    engine,
    harmful_train: list[dict],
    harmless_train: list[dict],
    pseudo_train: list[dict],
    harmful_valid: list[dict],
    harmless_valid: list[dict],
    pseudo_valid: list[dict],
    settings: dict,
    seed: int,
) -> dict:
    """Extract, validate and select one direction; inference remains weight-free."""
    n = int(settings["train_per_group"])
    v = int(settings["validation_per_group"])
    offsets = settings["prompt_token_offsets"]
    harmful = _select(engine, harmful_train, n, seed, require_refusal=True)
    harmless = _select(engine, harmless_train, n, seed + 1, require_refusal=False)
    pseudo = _select(engine, pseudo_train, n, seed + 2, require_refusal=True)
    h_val = _select(engine, harmful_valid, v, seed + 3, require_refusal=False)
    p_val = _select(engine, pseudo_valid, v, seed + 4, require_refusal=False)
    safe_val = _select(engine, harmless_valid, min(v, len(harmless_valid)), seed + 5, require_refusal=False)
    if not safe_val:
        raise ValueError("Need harmless validation prompts for the KL filter")

    true_candidates = _mean_hidden(engine, harmful, offsets) - _mean_hidden(engine, harmless, offsets)
    false_candidates = _mean_hidden(engine, pseudo, offsets) - _mean_hidden(engine, harmless, offsets)
    true_candidates = true_candidates.reshape(-1, true_candidates.shape[-1])
    false_candidates = false_candidates.reshape(-1, false_candidates.shape[-1])
    h_base = [engine.refusal_score(engine.first_logits(prompt)) for prompt in h_val]
    p_base = [engine.refusal_score(engine.first_logits(prompt)) for prompt in p_val]
    safe_logits = [engine.first_logits(prompt) for prompt in safe_val]

    best_true = max(
        range(len(true_candidates)),
        key=lambda index: _score_drop(engine, h_val, h_base, true_candidates[index]),
    )
    true_unit = true_candidates[best_true]
    best = None
    max_kl = float(settings["max_first_token_kl"])
    for index, candidate in enumerate(false_candidates):
        try:
            unit = orthogonalize(candidate, true_unit, float(settings["orthogonalization"]))
        except ValueError:
            continue
        kl = _first_token_kl(engine, safe_val, safe_logits, unit)
        if kl > max_kl:
            continue
        pseudo_drop = _score_drop(engine, p_val, p_base, unit)
        if best is None or pseudo_drop > best[0]:
            best = (pseudo_drop, index, unit, kl)
    if best is None:
        raise ValueError("No Surgical direction passed the first-token KL filter")
    pseudo_drop, selected, unit, kl = best
    return {
        "unit": unit.float().cpu(),
        "selected_layer": selected // len(offsets),
        "selected_token_offset": offsets[selected % len(offsets)],
        "true_layer": best_true // len(offsets),
        "true_token_offset": offsets[best_true % len(offsets)],
        "pseudo_refusal_score_drop": pseudo_drop,
        "harmful_refusal_score_drop": _score_drop(engine, h_val, h_base, unit),
        "first_token_kl": kl,
        "settings": json.loads(json.dumps(settings)),
        "sample_counts": {"harmful_train": n, "harmless_train": n, "pseudo_train": n, "validation": v},
    }


def save_vector(path: Path, artifact: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_vector(path: Path):
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact["unit"].ndim != 1 or not torch.isfinite(artifact["unit"]).all():
        raise ValueError(f"Invalid Surgical vector: {path}")
    return artifact
