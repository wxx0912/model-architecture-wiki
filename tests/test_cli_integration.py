from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PreflightCliIntegrationTest(unittest.TestCase):
    def run_preflight(self, fixture: str, *args: str) -> subprocess.CompletedProcess[str]:
        path = ROOT / "tests" / "fixtures" / "real" / fixture
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "preflight.py"),
                "--config",
                str(path),
                "--source-revision",
                "fixture-revision",
                "--compact",
                *args,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_mixtral_cli_identifies_moe_and_returns_checks_without_claiming_backend_support(self):
        result = self.run_preflight("mixtral_8x7b_instruct_config.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["profile_id"], "mixtral-8x7b-family")
        self.assertEqual(record["knowledge_page"], "paradigm-moe")
        self.assertEqual(record["structure"]["attention_layout"], "GQA")
        self.assertEqual(record["structure"]["num_experts"], 8)
        self.assertEqual(record["structure"]["num_experts_per_tok"], 2)
        checks = {item["feature"] for item in record["quantization_checks"]}
        self.assertTrue({"autoregressive", "gqa", "moe"} <= checks)
        self.assertEqual(record["quantization_constraints"]["status"], "not_assessed")
        self.assertEqual(record["quantization_constraints"]["joint_quantization_groups"], "unknown")

    def test_qwen2_vl_cli_extracts_vision_branch_and_multimodal_checks(self):
        result = self.run_preflight("qwen2_vl_7b_instruct_config.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["profile_id"], "qwen2-vl-family")
        self.assertEqual(record["structure"]["attention_layout"], "GQA")
        self.assertTrue(any("vision_config.depth=32" in path for path in record["structure"]["special_paths"]))
        checks = {item["feature"] for item in record["quantization_checks"]}
        self.assertTrue({"multimodal-vision", "mrope", "gqa"} <= checks)
        self.assertTrue(any(item["classification"] == "config-confirmed" for item in record["evidence"]))

    def test_strict_runtime_blocks_profile_level_structure_from_being_treated_as_ready(self):
        result = self.run_preflight("mixtral_8x7b_instruct_config.json", "--strict-runtime")
        self.assertEqual(result.returncode, 3)
        self.assertIn("strict runtime preflight blocked", result.stderr)
        self.assertIn("缺少 state_dict key 证据", result.stderr)
        self.assertIn("缺少 reference forward 证据", result.stderr)


if __name__ == "__main__":
    unittest.main()
