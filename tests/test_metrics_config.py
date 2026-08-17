#!/usr/bin/env python3

import json
import pathlib
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MetricsComposeTests(unittest.TestCase):
    def test_monitoring_images_are_version_pinned(self):
        compose = yaml.safe_load(
            (ROOT / "docker-compose.metrics.yaml").read_text(encoding="utf-8")
        )

        self.assertEqual(
            compose["services"]["prometheus"]["image"],
            "prom/prometheus:v3.13.2",
        )
        self.assertEqual(
            compose["services"]["grafana"]["image"],
            "grafana/grafana:13.1.3",
        )


class PerformanceDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = json.loads(
            (ROOT / "grafana/dashboards/vllm-throughput.json").read_text(
                encoding="utf-8"
            )
        )

    def test_panel_ids_are_unique_and_fit_the_grid(self):
        panels = self.dashboard["panels"]
        ids = [panel["id"] for panel in panels]

        self.assertEqual(len(ids), len(set(ids)))
        for panel in panels:
            grid = panel["gridPos"]
            self.assertGreater(grid["w"], 0)
            self.assertLessEqual(grid["x"] + grid["w"], 24)

    def test_all_panels_use_the_provisioned_prometheus_datasource(self):
        for panel in self.dashboard["panels"]:
            self.assertEqual(
                panel["datasource"],
                {"type": "prometheus", "uid": "prometheus"},
            )

    def test_latency_and_kv_cache_metrics_are_queried(self):
        expressions = {
            target["expr"]
            for panel in self.dashboard["panels"]
            for target in panel.get("targets", [])
        }
        rendered = "\n".join(expressions)

        for metric in (
            "vllm:time_to_first_token_seconds_bucket",
            "vllm:inter_token_latency_seconds_bucket",
            "vllm:e2e_request_latency_seconds_bucket",
            "vllm:kv_cache_usage_perc",
        ):
            self.assertIn(metric, rendered)


if __name__ == "__main__":
    unittest.main()
