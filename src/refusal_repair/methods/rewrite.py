"""Fixed prompt controls; these are comparison baselines, not the proposed method."""

from __future__ import annotations

from refusal_repair.refusal import is_refusal


def clarify(prompt: str) -> str:
    return (
        "Interpret the following request in its actual context. If it is safe, "
        "answer it directly and use the ordinary meaning of figurative or technical terms. "
        "If it requests harmful instructions, follow your normal safety policy.\n\n"
        f"Request: {prompt}"
    )


def run_rewrite(engine, prompt: str, conditional: bool) -> dict:
    first = engine.generate(prompt) if conditional else None
    if first is not None and not is_refusal(first["response"]):
        return {**first, "model_calls": 1, "triggered": False}
    second = engine.generate(clarify(prompt))
    return {
        **second,
        "model_calls": 2 if conditional else 1,
        "triggered": True,
        "elapsed_sec": round(second["elapsed_sec"] + (first or {}).get("elapsed_sec", 0), 4),
        "generated_tokens": second["generated_tokens"] + (first or {}).get("generated_tokens", 0),
    }
