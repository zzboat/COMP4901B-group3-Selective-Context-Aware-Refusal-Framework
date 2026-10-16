"""One-model-at-a-time inference shared by the local baselines."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path


class HFEngine:
    def __init__(self, model_path: Path, dtype: str = "float16", max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for the 8B baseline")
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self._refusal_ids = sorted({
            self.tokenizer.encode(word, add_special_tokens=False)[0]
            for word in ("I", "I'm", "Sorry", "As", "Unfortunately")
        })

    def input_ids(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)

    @contextlib.contextmanager
    def ablate(self, unit=None):
        if unit is None:
            yield
            return
        torch = self.torch
        unit = unit.to(self.device, dtype=self.model.dtype)
        unit = unit / unit.norm().clamp_min(1e-8)

        def hook(_module, _inputs, output):
            state = output[0] if isinstance(output, tuple) else output
            direction = unit.to(dtype=state.dtype)
            altered = state - (state * direction).sum(dim=-1, keepdim=True) * direction
            return (altered, *output[1:]) if isinstance(output, tuple) else altered

        handles = [layer.register_forward_hook(hook) for layer in self.model.model.layers]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def generate(self, prompt: str, unit=None) -> dict:
        torch = self.torch
        ids = self.input_ids(prompt)
        start = time.perf_counter()
        with torch.inference_mode(), self.ablate(unit):
            output = self.model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer_ids = output[0, ids.shape[1]:]
        return {
            "response": self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
            "generated_tokens": int(answer_ids.numel()),
            "elapsed_sec": round(time.perf_counter() - start, 4),
        }

    def first_logits(self, prompt: str, unit=None):
        torch = self.torch
        ids = self.input_ids(prompt)
        with torch.inference_mode(), self.ablate(unit):
            return self.model(ids, use_cache=False).logits[0, -1].float()

    def refusal_score(self, logits) -> float:
        torch = self.torch
        log_all = torch.logsumexp(logits, dim=-1)
        log_refusal = torch.logsumexp(logits[self._refusal_ids], dim=-1)
        probability = (log_refusal - log_all).exp().clamp(1e-8, 1 - 1e-8)
        return (probability.log() - torch.log1p(-probability)).item()

    def hidden_states(self, prompt: str, offsets: list[int]):
        torch = self.torch
        ids = self.input_ids(prompt)
        if any(abs(offset) > ids.shape[1] or offset >= 0 for offset in offsets):
            raise ValueError("Token offsets must be negative and inside the prompt")
        captured = []

        def capture(_module, _inputs, output):
            state = output[0] if isinstance(output, tuple) else output
            captured.append(torch.stack([state[0, offset].float().cpu() for offset in offsets]))

        handles = [layer.register_forward_hook(capture) for layer in self.model.model.layers]
        try:
            with torch.inference_mode():
                self.model(ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack(captured)
