from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preflight import (
    apply_runtime_config,
    apply_state_dict_keys,
    build_preflight,
    apply_forward_manifest,
    compact_preflight,
    load_profiles,
    load_forward_manifest,
    load_state_dict_keys,
    runtime_gaps,
    select_profile,
    validate_preflight,
)


class PreflightProfilesTest(unittest.TestCase):
    def test_every_profile_has_traceable_features_and_checks(self):
        profiles = load_profiles()
        self.assertGreaterEqual(len(profiles), 5)
        for profile in profiles:
            with self.subTest(profile=profile["id"]):
                result = build_preflight(profile)
                self.assertEqual(validate_preflight(result), [])
                self.assertEqual(result["structure"]["features"], profile["features"])
                self.assertEqual(len(result["evidence"]), len(profile["evidence"]))
                self.assertEqual(len(result["checklist"]), len(profile["features"]))
                self.assertTrue(all(item["source"].startswith(("paper-", "repo-")) for item in result["evidence"]))
                self.assertTrue(all(item["validation_status"] == "待实验验证" for item in result["checklist"]))
                self.assertEqual(result["implementation"]["revision"], "unknown")
                self.assertEqual(result["model"]["model_type"], "unknown")
                self.assertEqual(result["structure"]["num_hidden_layers"], "unknown")
                self.assertIn("module_dependencies", result["structure"])
                self.assertIn("special_paths", result["structure"])

    def test_profile_features_are_not_repeated_in_evidence(self):
        profiles = load_profiles()
        for profile in profiles:
            evidence_features = [item["feature"] for item in profile["evidence"]]
            self.assertEqual(len(evidence_features), len(set(evidence_features)), profile["id"])
            self.assertEqual(set(evidence_features), set(profile["features"]), profile["id"])

    def test_profiles_cover_requested_architecture_families(self):
        features = {feature for profile in load_profiles() for feature in profile["features"]}
        self.assertTrue({"autoregressive", "mha", "gqa", "mla", "moe", "diffusion-timestep", "masked-diffusion", "multimodal-vision", "linear-state"} <= features)

    def test_generated_record_fields_match_json_schema(self):
        schema = json.loads((ROOT / "data" / "preflight.schema.json").read_text(encoding="utf-8"))
        profile = load_profiles()[0]
        result = build_preflight(profile)
        self.assertEqual(set(result), set(schema["required"]))
        self.assertEqual(set(result["structure"]), set(schema["properties"]["structure"]["required"]))
        self.assertEqual(set(result["evidence"][0]), set(schema["$defs"]["evidence"]["required"]))
        self.assertEqual(set(result["checklist"][0]), set(schema["$defs"]["check"]["required"]))
        compact = compact_preflight(result)
        self.assertEqual(set(compact), set(schema["$defs"]["compact"]["required"]))

    def test_compact_view_preserves_runtime_evidence_and_marks_unassessed_constraints(self):
        profile = next(item for item in load_profiles() if item["id"] == "mixtral-8x7b-family")
        config_path = ROOT / "tests" / "fixtures" / "real" / "mixtral_8x7b_instruct_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        record = apply_runtime_config(
            build_preflight(profile), config, str(config_path.relative_to(ROOT)), model_type_hints=profile["model_type_hints"]
        )
        compact = compact_preflight(record)
        self.assertEqual(compact["knowledge_page"], "paradigm-moe")
        self.assertEqual(compact["structure"]["num_experts"], 8)
        self.assertIn("moe", {item["feature"] for item in compact["quantization_checks"]})
        self.assertTrue(all(item["classification"] in {"config-confirmed", "inferred"} for item in compact["evidence"]))
        self.assertEqual(compact["quantization_constraints"]["status"], "not_assessed")
        self.assertEqual(compact["quantization_constraints"]["supported_formats_and_kernels"], "unknown")
        self.assertTrue(compact["omitted_profile_evidence_count"] > 0)

    def test_compact_profile_only_keeps_source_evidence(self):
        record = build_preflight(load_profiles()[0])
        compact = compact_preflight(record)
        self.assertEqual(compact["omitted_profile_evidence_count"], 0)
        self.assertTrue(compact["evidence"])

    def test_runtime_config_fields_are_observed_and_inferences_are_labeled(self):
        profile = next(item for item in load_profiles() if item["id"] == "qwen2-vl-family")
        config_path = ROOT / "tests" / "fixtures" / "qwen2_vl_config.json"
        record = apply_runtime_config(
            build_preflight(profile),
            json.loads(config_path.read_text(encoding="utf-8")),
            str(config_path.relative_to(ROOT)),
            source_revision="fixture-revision",
            model_type_hints=profile["model_type_hints"],
        )
        self.assertEqual(validate_preflight(record), [])
        self.assertEqual(record["model"]["model_type"], "qwen2_vl")
        self.assertEqual(record["model"]["architectures"], ["Qwen2VLForConditionalGeneration"])
        self.assertEqual(record["evidence"][0]["version_or_revision"], "unknown")
        self.assertEqual(record["structure"]["num_hidden_layers"], 32)
        self.assertEqual(record["structure"]["attention_layout"], "GQA")
        self.assertIn("gqa", {item["feature"] for item in record["checklist"]})
        self.assertEqual(record["structure"]["parameter_sharing"], "tie_word_embeddings=false")
        self.assertIn("config.vision_config.depth=32", record["structure"]["special_paths"])
        self.assertTrue(any(item["claim"] == "vision_config:depth" for item in record["evidence"]))
        layout_evidence = next(item for item in record["evidence"] if item["claim"] == "attention_layout")
        self.assertEqual(layout_evidence["classification"], "inferred")
        self.assertIn("text_config.num_attention_heads", layout_evidence["locator"])
        self.assertEqual(next(item for item in record["evidence"] if item["claim"] == "num_hidden_layers")["classification"], "config-confirmed")
        self.assertTrue(all(item["validation_status"] == "待实验验证" for item in record["checklist"]))

    def test_nested_model_type_is_checked_and_located(self):
        profile = next(item for item in load_profiles() if item["id"] == "qwen2-vl-family")
        config = {"text_config": {"model_type": "qwen2_vl", "num_hidden_layers": 2}}
        record = apply_runtime_config(
            build_preflight(profile), config, "nested-config.json", model_type_hints=profile["model_type_hints"]
        )
        self.assertEqual(record["model"]["model_type"], "qwen2_vl")
        self.assertEqual(record["structure"]["num_hidden_layers"], 2)
        self.assertEqual(
            next(item for item in record["evidence"] if item["claim"] == "model_type")["locator"],
            "nested-config.json#/text_config.model_type",
        )
        with self.assertRaisesRegex(ValueError, "does not match profile hints"):
            apply_runtime_config(
                build_preflight(profile),
                {"text_config": {"model_type": "other"}},
                "nested-config.json",
                model_type_hints=profile["model_type_hints"],
            )

    def test_moe_fixture_extracts_expert_layout_and_top_k(self):
        profile = next(item for item in load_profiles() if item["id"] == "mixtral-8x7b-family")
        config_path = ROOT / "tests" / "fixtures" / "mixtral_moe_config.json"
        record = apply_runtime_config(
            build_preflight(profile),
            json.loads(config_path.read_text(encoding="utf-8")),
            str(config_path.relative_to(ROOT)),
            model_type_hints=profile["model_type_hints"],
        )
        self.assertEqual(validate_preflight(record), [])
        self.assertEqual(record["structure"]["num_experts"], 8)
        self.assertEqual(record["structure"]["num_experts_per_tok"], 2)
        self.assertEqual(record["structure"]["moe_layout"], "MoE: 8 experts; top_k=2")
        self.assertIn("moe", record["structure"]["features"])

    def test_gqa_fixture_adds_matching_quantization_check(self):
        profile = next(item for item in load_profiles() if item["id"] == "attention-variants-reference")
        config_path = ROOT / "tests" / "fixtures" / "qwen2_gqa_config.json"
        record = apply_runtime_config(
            build_preflight(profile),
            json.loads(config_path.read_text(encoding="utf-8")),
            str(config_path.relative_to(ROOT)),
        )
        checks = {item["feature"] for item in record["checklist"]}
        self.assertEqual(record["structure"]["attention_layout"], "GQA")
        self.assertIn("gqa", checks)

    def test_dit_and_dllm_config_fixtures_select_the_expected_profile(self):
        profiles = load_profiles()
        for fixture_name, expected_profile in (("dit_config.json", "dit-xl-2-reference"), ("llada_config.json", "llada-family")):
            with self.subTest(fixture=fixture_name):
                path = ROOT / "tests" / "fixtures" / fixture_name
                config = json.loads(path.read_text(encoding="utf-8"))
                profile = select_profile(profiles, None, config)
                self.assertEqual(profile["id"], expected_profile)
                result = apply_runtime_config(build_preflight(profile), config, str(path.relative_to(ROOT)), model_type_hints=profile["model_type_hints"])
                self.assertEqual(validate_preflight(result), [])
                if expected_profile == "dit-xl-2-reference":
                    self.assertEqual(result["structure"]["attention_layout"], "self-attention; exact block layout unknown")

    def test_real_config_snapshots_select_expected_profiles(self):
        profiles = load_profiles()
        cases = (
            ("qwen2_7b_instruct_config.json", "qwen2-decoder-family", "GQA", 28, 4),
            ("mixtral_8x7b_instruct_config.json", "mixtral-8x7b-family", "GQA", 32, 8),
            ("qwen2_vl_7b_instruct_config.json", "qwen2-vl-family", "GQA", 28, 4),
        )
        for filename, expected_profile, expected_layout, expected_layers, expected_kv_heads in cases:
            with self.subTest(filename=filename):
                path = ROOT / "tests" / "fixtures" / "real" / filename
                config = json.loads(path.read_text(encoding="utf-8"))
                profile = select_profile(profiles, None, config)
                self.assertEqual(profile["id"], expected_profile)
                result = apply_runtime_config(
                    build_preflight(profile),
                    config,
                    str(path.relative_to(ROOT)),
                    source_revision="snapshot-config-commit",
                    model_type_hints=profile["model_type_hints"],
                )
                self.assertEqual(validate_preflight(result), [])
                self.assertEqual(result["structure"]["attention_layout"], expected_layout)
                self.assertEqual(result["structure"]["num_hidden_layers"], expected_layers)
                self.assertEqual(result["structure"]["num_key_value_heads"], expected_kv_heads)

        mamba_path = ROOT / "tests" / "fixtures" / "real" / "mamba2_130m_config.json"
        mamba_config = json.loads(mamba_path.read_text(encoding="utf-8"))
        mamba_profile = select_profile(profiles, None, mamba_config)
        self.assertEqual(mamba_profile["id"], "mamba2-family")
        mamba_record = apply_runtime_config(
            build_preflight(mamba_profile),
            mamba_config,
            str(mamba_path.relative_to(ROOT)),
            source_revision="snapshot-config-commit",
            config_signatures=mamba_profile["config_signatures"],
        )
        self.assertEqual(validate_preflight(mamba_record), [])
        self.assertEqual(mamba_record["structure"]["num_hidden_layers"], 24)
        self.assertEqual(mamba_record["structure"]["hidden_size"], 768)
        self.assertTrue(any(item["claim"] == "architecture_signature" for item in mamba_record["evidence"]))

    def test_profile_hint_rejects_mismatched_model_type(self):
        profile = next(item for item in load_profiles() if item["id"] == "mixtral-8x7b-family")
        with self.assertRaisesRegex(ValueError, "does not match profile hints"):
            apply_runtime_config(build_preflight(profile), {"model_type": "qwen2"}, "config.json", model_type_hints=profile["model_type_hints"])

    def test_config_can_select_a_unique_profile(self):
        profile = select_profile(load_profiles(), None, {"model_type": "mixtral"})
        self.assertEqual(profile["id"], "mixtral-8x7b-family")
        qwen_profile = select_profile(load_profiles(), None, {"model_type": "qwen2"})
        self.assertEqual(qwen_profile["id"], "qwen2-decoder-family")

    def test_runtime_config_does_not_fill_missing_fields_with_guesses(self):
        profile = next(
            item for item in load_profiles()
            if item["id"] == "attention-variants-reference"
        )
        record = apply_runtime_config(build_preflight(profile), {"model_type": "custom"}, "config.json", model_type_hints=profile["model_type_hints"])
        self.assertEqual(record["structure"]["num_hidden_layers"], "unknown")
        self.assertEqual(record["structure"]["hidden_size"], "unknown")
        self.assertEqual(record["structure"]["attention_layout"], "MHA, GQA or MLA; resolve from target config and implementation")
        self.assertIn("Attention 布局未从配置/运行图解析", runtime_gaps(record))

    def test_state_dict_key_manifest_confirms_only_names_and_contiguous_layers(self):
        profile = next(item for item in load_profiles() if item["id"] == "qwen2-decoder-family")
        path = ROOT / "tests" / "fixtures" / "qwen2_state_dict_keys.json"
        record = apply_state_dict_keys(
            build_preflight(profile),
            load_state_dict_keys(path),
            str(path.relative_to(ROOT)),
            complete=True,
        )
        self.assertEqual(validate_preflight(record), [])
        self.assertEqual(record["structure"]["num_hidden_layers"], 2)
        self.assertIn("state-dict:attention", record["structure"]["special_paths"])
        self.assertIn("attention", record["structure"]["module_dependencies"])
        self.assertNotIn("moe", record["structure"]["features"])
        self.assertTrue(any(item["classification"] == "state-dict-confirmed" for item in record["evidence"]))
        self.assertEqual(record["structure"]["hidden_size"], "unknown")

    def test_hf_weight_map_manifest_detects_experts(self):
        profile = next(item for item in load_profiles() if item["id"] == "mixtral-8x7b-family")
        path = ROOT / "tests" / "fixtures" / "mixtral_state_dict_index.json"
        record = apply_state_dict_keys(
            build_preflight(profile),
            load_state_dict_keys(path),
            str(path.relative_to(ROOT)),
            complete=True,
        )
        self.assertEqual(validate_preflight(record), [])
        self.assertEqual(record["structure"]["num_hidden_layers"], 2)
        self.assertEqual(record["structure"]["num_experts"], 2)
        self.assertIn("moe", record["structure"]["features"])

    def test_state_dict_conflict_is_rejected(self):
        profile = next(item for item in load_profiles() if item["id"] == "mixtral-8x7b-family")
        record = build_preflight(profile)
        record["structure"]["num_hidden_layers"] = 4
        with self.assertRaisesRegex(ValueError, "exceeds"):
            apply_state_dict_keys(
                record,
                [f"model.layers.{index}.weight" for index in range(5)],
                "keys.txt",
                complete=False,
            )
        with self.assertRaisesRegex(ValueError, "complete state-dict layer count"):
            conflict_record = build_preflight(profile)
            conflict_record["structure"]["num_hidden_layers"] = 32
            apply_state_dict_keys(
                conflict_record,
                ["model.layers.0.weight", "model.layers.1.weight"],
                "keys.txt",
                complete=True,
            )

    def test_incomplete_state_dict_manifest_does_not_fill_unknown_counts(self):
        profile = next(item for item in load_profiles() if item["id"] == "qwen2-decoder-family")
        record = apply_state_dict_keys(
            build_preflight(profile),
            ["model.layers.0.weight", "model.layers.1.weight"],
            "partial-keys.txt",
        )
        self.assertEqual(record["structure"]["num_hidden_layers"], "unknown")
        self.assertTrue(any(item["claim"] == "layer_index_coverage" for item in record["evidence"]))

    def test_forward_manifest_records_observed_tensor_contract(self):
        profile = next(item for item in load_profiles() if item["id"] == "qwen2-decoder-family")
        path = ROOT / "tests" / "fixtures" / "forward_manifest.json"
        observations, metadata = load_forward_manifest(path)
        record = apply_forward_manifest(
            build_preflight(profile),
            observations,
            str(path.relative_to(ROOT)),
            metadata,
        )
        self.assertEqual(validate_preflight(record), [])
        self.assertIn("model.layers.0.self_attn.q_proj", record["structure"]["tensor_flow"])
        self.assertIn("bfloat16", record["structure"]["tensor_flow"])
        self.assertEqual(
            sum(item["classification"] == "forward-confirmed" for item in record["evidence"]),
            7,
        )
        self.assertTrue(any(item["claim"] == "forward_metadata:device" for item in record["evidence"]))

    def test_strict_runtime_blocks_profile_only_and_accepts_complete_runtime_evidence(self):
        profile = next(item for item in load_profiles() if item["id"] == "qwen2-decoder-family")
        self.assertIn("checkpoint 未锁定", runtime_gaps(build_preflight(profile)))
        config_path = ROOT / "tests" / "fixtures" / "qwen2_gqa_config.json"
        key_path = ROOT / "tests" / "fixtures" / "qwen2_state_dict_keys.json"
        forward_path = ROOT / "tests" / "fixtures" / "forward_manifest.json"
        record = apply_runtime_config(
            build_preflight(profile),
            json.loads(config_path.read_text(encoding="utf-8")),
            str(config_path.relative_to(ROOT)),
            source_revision="fixture-revision",
            model_type_hints=profile["model_type_hints"],
        )
        record = apply_state_dict_keys(record, load_state_dict_keys(key_path), str(key_path.relative_to(ROOT)))
        observations, metadata = load_forward_manifest(forward_path)
        record = apply_forward_manifest(record, observations, str(forward_path.relative_to(ROOT)), metadata)
        self.assertEqual(runtime_gaps(record), [])

    def test_strict_runtime_requires_forward_provenance(self):
        profile = next(item for item in load_profiles() if item["id"] == "qwen2-decoder-family")
        config_path = ROOT / "tests" / "fixtures" / "qwen2_gqa_config.json"
        key_path = ROOT / "tests" / "fixtures" / "qwen2_state_dict_keys.json"
        forward_path = ROOT / "tests" / "fixtures" / "forward_manifest.json"
        record = apply_runtime_config(
            build_preflight(profile),
            json.loads(config_path.read_text(encoding="utf-8")),
            str(config_path.relative_to(ROOT)),
            source_revision="fixture-revision",
            model_type_hints=profile["model_type_hints"],
        )
        record = apply_state_dict_keys(record, load_state_dict_keys(key_path), str(key_path.relative_to(ROOT)))
        observations, _ = load_forward_manifest(forward_path)
        record = apply_forward_manifest(record, observations, str(forward_path.relative_to(ROOT)), {})
        gaps = runtime_gaps(record)
        self.assertIn("reference forward 元数据 source_revision 未记录", gaps)
        self.assertIn("reference forward 元数据 input_signature 未记录", gaps)


if __name__ == "__main__":
    unittest.main()
