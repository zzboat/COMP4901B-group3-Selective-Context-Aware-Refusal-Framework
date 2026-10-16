from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from refusal_repair.datasets import file_sha256, load_split, prepare_splits  # noqa: E402
from refusal_repair.evaluation import label_response, summarize  # noqa: E402
from refusal_repair.methods.rewrite import run_rewrite  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare import paired_result  # noqa: E402


class SplitTests(unittest.TestCase):
    def test_fixed_counts_balance_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw"
            raw.mkdir(parents=True)

            def save_csv(name, rows, columns):
                with (raw / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)

            save_csv("xstest_prompts.csv", [
                {"id": i, "prompt": f"xstest prompt {i}", "label": "safe" if i <= 250 else "unsafe"}
                for i in range(1, 451)
            ], ["id", "prompt", "label"])
            save_csv("or-bench-hard-1k.csv", [
                {"prompt": f"hard prompt {i}", "category": "test"} for i in range(1319)
            ], ["prompt", "category"])
            save_csv("or-bench-toxic.csv", [
                {"prompt": f"toxic prompt {i}", "category": "test"} for i in range(655)
            ], ["prompt", "category"])
            (raw / "sorry-bench-202503").mkdir()
            with (raw / "sorry-bench-202503/question.jsonl").open("w", encoding="utf-8") as handle:
                for category in range(1, 45):
                    for index in range(10):
                        handle.write(json.dumps({
                            "question_id": category * 10 + index,
                            "category": category,
                            "turns": [f"sorry prompt {category}-{index}"],
                        }) + "\n")
            entries = [
                {"name": name, "path": str(path.relative_to(root)), "sha256": file_sha256(path)}
                for name, path in [
                    ("xstest", raw / "xstest_prompts.csv"),
                    ("hard", raw / "or-bench-hard-1k.csv"),
                    ("toxic", raw / "or-bench-toxic.csv"),
                    ("sorry", raw / "sorry-bench-202503/question.jsonl"),
                ]
            ]
            (root / "data/manifest.json").write_text(json.dumps({"datasets": entries}))
            manifest = prepare_splits(root, seed=42)
            self.assertEqual([manifest["splits"][name]["rows"] for name in (
                "or_hard_extract", "or_hard_valid", "or_hard_test", "or_toxic_test",
                "xstest_safe_test", "xstest_unsafe_test", "sorry_test",
            )], [300, 200, 500, 600, 250, 200, 220])
            sorry = load_split(root, "sorry_test")
            self.assertEqual({sum(row["category"] == str(category) for row in sorry) for category in range(1, 45)}, {5})
            first_hash = manifest["splits"]["or_hard_test"]["sha256"]
            self.assertEqual(prepare_splits(root, seed=42)["splits"]["or_hard_test"]["sha256"], first_hash)


class RewriteAndScoreTests(unittest.TestCase):
    def test_retry_only_after_refusal(self):
        class Engine:
            def __init__(self):
                self.calls = []

            def generate(self, prompt):
                self.calls.append(prompt)
                return {"response": "I cannot help" if len(self.calls) == 1 else "Here is the answer", "elapsed_sec": 1.0, "generated_tokens": 4}

        engine = Engine()
        result = run_rewrite(engine, "How do I kill a Python process?", conditional=True)
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["elapsed_sec"], 2.0)
        self.assertIn("Python process", engine.calls[1])

    def test_summary_separates_safe_and_unsafe(self):
        rows = [
            label_response({"method": "original", "dataset": "xstest", "label": "safe", "response": "I cannot help"}),
            label_response({"method": "original", "dataset": "xstest", "label": "safe", "response": "Use the switch"}),
            label_response({"method": "original", "dataset": "sorry", "label": "unsafe", "response": "I cannot help"}, False),
        ]
        result = summarize(rows)
        safe = next(row for row in result if row["label"] == "safe")
        self.assertEqual(safe["refusal_rate"], 0.5)
        unsafe = next(row for row in result if row["label"] == "unsafe")
        self.assertEqual(unsafe["guard_unsafe_answer_rate"], 0)

    def test_paired_comparison_rejects_mismatched_ids(self):
        base = [{"id": "a", "refusal_heuristic": True}]
        changed = [{"id": "b", "refusal_heuristic": False}]
        with self.assertRaises(ValueError):
            paired_result(base, changed, "refusal_heuristic")


class SurgicalMathTests(unittest.TestCase):
    def test_orthogonalization(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from refusal_repair.methods.surgical import orthogonalize

        true = torch.tensor([1.0, 0.0])
        false = torch.tensor([1.0, 1.0])
        result = orthogonalize(false, true)
        self.assertAlmostEqual(float(result @ true), 0.0, places=6)
        self.assertAlmostEqual(float(result.norm()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
