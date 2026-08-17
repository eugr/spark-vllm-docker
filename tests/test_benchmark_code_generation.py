#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_code_generation", ROOT / "benchmark-code-generation.py"
)
assert SPEC is not None and SPEC.loader is not None
CODE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CODE)


class SuiteTests(unittest.TestCase):
    def test_default_suite_is_valid_and_unique(self):
        tasks = CODE.validate_tasks(CODE.DEFAULT_TASKS)

        self.assertEqual(len(tasks), 4)
        self.assertEqual(len({task["name"] for task in tasks}), 4)

    def test_duplicate_task_names_are_rejected(self):
        tasks = [
            {"name": "same", "language": "Python", "prompt": "one"},
            {"name": "same", "language": "Rust", "prompt": "two"},
        ]

        with self.assertRaises(CODE.CORE.BenchmarkError):
            CODE.validate_tasks(tasks)


class AggregateTests(unittest.TestCase):
    def test_aggregates_tasks_without_treating_distinct_prompts_as_instability(self):
        task_results = []
        for name, rate, digest in (("python", 20.0, "a"), ("rust", 18.0, "b")):
            runs = [
                {
                    "decode_tokens_per_second": rate,
                    "ttft_seconds": 0.2,
                    "total_seconds": 26.0,
                    "completion_tokens": 512,
                    "response_sha256": digest,
                }
                for _ in range(2)
            ]
            task_results.append(
                {"name": name, "runs": runs, "summary": CODE.CORE.summarize_runs(runs)}
            )

        summary = CODE.aggregate_task_results(task_results)

        self.assertEqual(summary["median_decode_tokens_per_second"], 19.0)
        self.assertTrue(summary["outputs_stable_within_config"])
        self.assertEqual(summary["response_hashes"], {"python": "a", "rust": "b"})

    def test_skips_runs_without_a_decode_rate(self):
        runs = [
            {
                "decode_tokens_per_second": rate,
                "ttft_seconds": 0.2,
                "total_seconds": 26.0,
                "completion_tokens": 512,
                "response_sha256": "a",
            }
            for rate in (20.0, 18.0, 19.0, None)
        ]
        task_results = [
            {"name": "python", "runs": runs, "summary": CODE.CORE.summarize_runs(runs)}
        ]

        summary = CODE.aggregate_task_results(task_results)

        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["median_decode_tokens_per_second"], 19.0)
        self.assertTrue(summary["outputs_stable_within_config"])

    def test_raises_when_no_decode_rate_can_be_calculated(self):
        runs = [
            {
                "decode_tokens_per_second": None,
                "ttft_seconds": 0.2,
                "total_seconds": 26.0,
                "completion_tokens": 512,
                "response_sha256": "a",
            }
        ]
        task_results = [
            {
                "name": "python",
                "runs": runs,
                "summary": {
                    "outputs_stable_within_config": True,
                    "response_sha256": "a",
                },
            }
        ]

        with self.assertRaises(CODE.CORE.BenchmarkError):
            CODE.aggregate_task_results(task_results)


if __name__ == "__main__":
    unittest.main()
