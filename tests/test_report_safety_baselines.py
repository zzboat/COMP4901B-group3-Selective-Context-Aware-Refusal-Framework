"""Checks for paired safety reporting, without loading a model."""

import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/report_safety_baselines.py"
SPEC = importlib.util.spec_from_file_location("report_safety_baselines", PATH)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def row(key, unsafe, method="original"):
    return {"id": key, "prompt": key, "dataset": "fixture", "label": "unsafe",
            "method": method, "guard_unsafe_answer": unsafe, "refusal_heuristic": not unsafe}


class ReportTests(unittest.TestCase):
    def test_rejects_missing_judge_and_mismatched_prompts(self):
        original = [row("a", False)]
        for field, value in [("guard_unsafe_answer", None), ("prompt", "different"), ("label", "safe")]:
            changed = [{**original[0], field: value}]
            with self.subTest(field=field), self.assertRaises(ValueError):
                report.validate_rows(changed, original, "original", 1)

    def test_rejects_duplicate_ids(self):
        rows = [row("a", False), row("a", True)]
        with self.assertRaises(ValueError):
            report.validate_rows(rows, rows, "original", 2)

    def test_equal_rates_still_report_new_and_removed_failures(self):
        data = {}
        for split in report.SPLITS:
            for method in report.METHODS:
                data[(split, method)] = [row("a", method != "original", method),
                                        row("b", method == "original", method)]
        text = report.build_report(data, {"fixture": "abc"}, {"test-commit"})
        self.assertIn("| xstest_unsafe_test | surgical | 1 | 1 | +0 |", text)
        self.assertIn("1 / 50.00%", text)
        self.assertIn("not the official SORRY-Bench", text)


if __name__ == "__main__":
    unittest.main()
