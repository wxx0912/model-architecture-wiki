#!/usr/bin/env python3
"""Generate an evidence-aware quantization preflight from a curated profile."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from wiki_common import ROOT, pages as pages_of, sources

PROFILE_FILE = ROOT / "data" / "architecture-profiles.json"

CHECKS = {
    "autoregressive": (
        "prefill/decode 激活与 cache 误差可能不同",
        "按短/长上下文分别比较 prefill、decode、cache 读写输出",
        ("paradigm-ar",),
    ),
    "mla": (
        "latent 压缩、恢复和 latent cache 的误差路径不同于普通 K/V cache",
        "从目标实现记录 latent 维度、cache layout，并比较 attention 输出与长上下文结果",
        ("component-attention", "model-deepseek"),
    ),
    "moe": (
        "router 微小误差可能改变 top-k，专家覆盖也可能不均衡",
        "记录量化前后 token-to-expert 映射、专家访问直方图和专家输出误差",
        ("paradigm-moe", "component-router-expert"),
    ),
    "diffusion-timestep": (
        "不同 timestep/noise level 可能产生不同激活分布",
        "沿实际采样轨迹分层采集早、中、晚阶段激活和逐步输出",
        ("component-diffusion-timestep", "paradigm-dit"),
    ),
    "attention": (
        "投影、softmax 与输出路径的误差可能不同",
        "按目标 attention 实现比较投影输出、attention 输出和最终 logits",
        ("component-attention",),
    ),
    "rope": (
        "旋转位置编码的相位与频率若在量化或融合中改变，注意力打分随之变化",
        "核对 theta、维度切分方式与 cache 中是否已应用，比较旋转前后 Q/K 与 attention 输出",
        ("component-position-encoding",),
    ),
    "rmsnorm": (
        "归一化统计与 scale 的融合边界会改变后续激活分布",
        "记录 epsilon、是否与相邻线性层融合及量化位置，比较 norm 输出与紧邻投影误差",
        ("component-normalization",),
    ),
    "swiglu": (
        "gate/up 与激活的取值范围不同，统一缩放可能让其中一路过冲",
        "记录 gate/up/down 的拆分与融合方式，分别比较 gate、激活后与 down 输出误差",
        ("component-mlp",),
    ),
    "residual": (
        "跨层残差累加会放大逐层舍入误差",
        "核对残差相加位置与写回精度，比较逐层 hidden state 与最终输出误差",
        ("component-residual-mhc", "component-attention-residuals"),
    ),
    "mha": (
        "各 query head 有独立 K/V 路径，统一缩放可能掩盖 head 间分布差异",
        "记录 q/kv head 数与 head_dim，比较各投影和 attention 输出误差",
        ("component-attention",),
    ),
    "gqa": (
        "多个 query head 共享 KV head，量化误差可能在共享路径上被放大",
        "确认 q_heads/kv_heads 映射、cache shape/layout，并测量各 query group 输出差异",
        ("component-attention", "component-kv-cache"),
    ),
    "masked-diffusion": (
        "logit 误差可能改变位置置信度排序和后续 mask 轨迹",
        "固定 sampler，逐步比较 logits、位置选择、mask 状态和最终输出",
        ("paradigm-dllm", "component-diffusion-timestep"),
    ),
    "autoregressive-transformer-backbone": (
        "Transformer 主干存在多条独立投影与残差误差路径",
        "以目标 checkpoint 的 module tree、state dict 和 reference forward 核对模块边界",
        ("paradigm-ar",),
    ),
    "multimodal-vision": (
        "视觉编码器、connector 与语言主干的分布可能不同",
        "按分辨率/帧数分层检查视觉 token、connector 输出和端到端多模态结果",
        ("component-multimodal-connector", "paradigm-multimodal"),
    ),
    "mrope": (
        "多模态位置与时间/空间索引错误可能改变跨模态注意力行为",
        "核对 position ids、模态边界、图像 grid/视频时间索引及 attention mask",
        ("component-position-encoding", "model-qwen2-vl"),
    ),
    "linear-state": (
        "递推 state 的舍入误差可能随序列累积，并在 chunk/reset 边界变化",
        "记录 state dtype/layout，比较短长序列、chunk 边界、连续 decode 和 reset 输出",
        ("paradigm-ssm", "paradigm-linear-attention"),
    ),
}


def make_checklist(features: list[str]) -> list[dict[str, object]]:
    """把一个结构特征转成“该读哪些页面 + 待核实项”，不产出候选方案。"""
    checklist = []
    for feature in features:
        if feature not in CHECKS:
            raise ValueError(f"no preflight rule for feature: {feature}")
        risk, method, pages = CHECKS[feature]
        checklist.append({
            "feature": feature,
            "pages": list(pages),
            "potential_risk": risk,
            "check_method": method,
            "validation_status": "待核实",
        })
    return checklist


def load_profiles() -> list[dict[str, object]]:
    payload = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("profiles"), list):
        raise ValueError("unsupported architecture profile registry")
    return payload["profiles"]


def select_profile(
    profiles: list[dict[str, object]],
    profile_id: str | None,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    if profile_id:
        profile = next((item for item in profiles if item["id"] == profile_id), None)
        if profile is None:
            raise ValueError(f"profile not found: {profile_id}")
        return profile
    if config is None:
        raise ValueError("provide a profile ID or --config for automatic profile selection")
    nested = config.get("text_config")
    model_type = config.get("model_type")
    if not isinstance(model_type, str) and isinstance(nested, dict):
        model_type = nested.get("model_type")
    matches = [item for item in profiles if model_type in item.get("model_type_hints", [])]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"model_type {model_type!r} matches multiple profiles; pass a profile ID")
    signature_matches = []
    for item in profiles:
        for signature in item.get("config_signatures", []):
            value = config
            for part in str(signature.get("path", "")).split("."):
                if not isinstance(value, dict) or part not in value:
                    value = None
                    break
                value = value[part]
            if value == signature.get("equals"):
                signature_matches.append(item)
                break
    if len(signature_matches) == 1:
        return signature_matches[0]
    if len(signature_matches) > 1:
        raise ValueError("config matches multiple architecture signatures; pass a profile ID")
    raise ValueError(
        f"no profile matches model_type {model_type!r}; the profile registry is frozen (page routing only) — "
        "pass an explicit profile ID (see --list) instead of adding a new profile"
    )


def build_preflight(profile: dict[str, object]) -> dict[str, object]:
    registry = sources(ROOT)
    evidence = []
    for item in profile["evidence"]:
        source_id = item["source"]
        if source_id not in registry:
            raise ValueError(f"unknown source ID: {source_id}")
        source = registry[source_id]
        evidence.append({
            "claim": item["feature"],
            "classification": item["classification"],
            "source": f"{source_id} ({source.meta['url']})",
            "version_or_revision": item.get("version_or_revision", "unknown") if source.meta["category"] == "paper" else profile["implementation"]["revision"],
            "locator": item["locator"],
        })
    checklist = make_checklist(profile["features"])
    return {
        "schema_version": 1,
        "profile_id": profile["id"],
        "knowledge_page": profile["knowledge_page"],
        "model": {
            "name": profile["model_name"],
            "checkpoint": profile["checkpoint"],
            "model_type": profile["model_type"],
            "architectures": profile["architectures"],
        },
        "implementation": profile["implementation"],
        "runtime": {
            "source_revision": "unknown",
            "device": "unknown",
            "dtype": "unknown",
            "input_signature": "unknown",
        },
        "structure": {
            "features": list(profile["features"]),
            **profile["structure"],
            "num_attention_heads": "unknown",
            "num_key_value_heads": "unknown",
            "num_experts": "unknown",
            "num_experts_per_tok": "unknown",
            "tensor_flow": "需从目标 checkpoint reference forward 采集并记录 shape/dtype/layout",
        },
        "evidence": evidence,
        "checklist": checklist,
        "required_runtime_confirmation": [
            "锁定 checkpoint、model_type、architectures 与配置字段",
            "锁定源码仓库 commit/tag；profile 中的 unknown 不得沿用为模型事实",
            "补全层数、维度、模块依赖、共享参数、融合关系和特殊计算路径",
            "对每个结构字段记录来源版本、代码路径或运行图证据",
        ],
    }


def quantization_view(record: dict[str, object]) -> dict[str, object]:
    """Return the quantization-stage view intended for Agent context."""
    runtime_classes = {"config-confirmed", "state-dict-confirmed", "forward-confirmed", "inferred"}
    runtime_evidence = [item for item in record["evidence"] if item["classification"] in runtime_classes]
    selected_evidence = runtime_evidence or record["evidence"]
    evidence = []
    for item in selected_evidence:
        evidence.append({
            "claim": item["claim"],
            "classification": item["classification"],
            "source": item["source"],
            "version_or_revision": item["version_or_revision"],
            "locator": item["locator"],
        })
    return {
        "schema_version": record["schema_version"],
        "profile_id": record["profile_id"],
        "knowledge_page": record["knowledge_page"],
        "model": record["model"],
        "implementation": record["implementation"],
        "runtime": record["runtime"],
        "structure": record["structure"],
        "evidence": evidence,
        "omitted_profile_evidence_count": len(record["evidence"]) - len(evidence),
        "quantization_checks": record["checklist"],
        "quantization_constraints": {
            "status": "not_assessed",
            "quantizable_modules": "unknown",
            "joint_quantization_groups": "unknown",
            "excluded_modules": "unknown",
            "supported_formats_and_kernels": "unknown",
        },
        "runtime_gaps": runtime_gaps(record),
    }


def apply_runtime_config(
    record: dict[str, object],
    config: dict[str, object],
    config_path: str,
    checkpoint: str | None = None,
    source_revision: str | None = None,
    model_type_hints: list[str] | None = None,
    config_signatures: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Overlay directly observed HF config fields and record their provenance."""
    if not isinstance(config, dict):
        raise ValueError("model config JSON root must be an object")
    nested = config.get("text_config")
    model_type = config.get("model_type")
    if not isinstance(model_type, str) and isinstance(nested, dict):
        model_type = nested.get("model_type")
    hints = model_type_hints or []
    if hints and isinstance(model_type, str) and model_type not in hints:
        raise ValueError(f"config model_type {model_type!r} does not match profile hints {hints}")
    result = copy.deepcopy(record)
    effective = nested if isinstance(nested, dict) else config
    nested_prefix = "text_config." if isinstance(nested, dict) else ""
    model = result["model"]
    structure = result["structure"]
    implementation = result["implementation"]
    evidence = result["evidence"]

    for signature in config_signatures or []:
        value = config
        path = str(signature.get("path", ""))
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value == signature.get("equals"):
            evidence.append({
                "claim": "architecture_signature",
                "classification": "config-confirmed",
                "source": "runtime-config",
                "version_or_revision": str(source_revision or "config snapshot; code revision unknown"),
                "locator": f"{config_path}#/{path}",
            })

    def observed(field: str, keys: tuple[str, ...], default: object = "unknown") -> object:
        for key in keys:
            if key in effective and effective[key] is not None:
                value = effective[key]
                if field in {"num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "num_experts", "num_experts_per_tok"} and not isinstance(value, (int, str)):
                    raise ValueError(f"config field {nested_prefix}{key} must be an integer or 'unknown'")
                evidence.append({
                    "claim": field,
                    "classification": "config-confirmed",
                    "source": "runtime-config",
                    "version_or_revision": str(source_revision or "config snapshot; code revision unknown"),
                    "locator": f"{config_path}#/{nested_prefix}{key}",
                })
                return value
        return default

    model_type_key = "model_type" if isinstance(config.get("model_type"), str) else f"{nested_prefix}model_type"
    model_type = config.get("model_type") if isinstance(config.get("model_type"), str) else effective.get("model_type")
    if isinstance(model_type, str) and model_type:
        model["model_type"] = model_type
        evidence.append({
            "claim": "model_type",
            "classification": "config-confirmed",
            "source": "runtime-config",
            "version_or_revision": str(source_revision or "config snapshot; code revision unknown"),
            "locator": f"{config_path}#/{model_type_key}",
        })
    architectures = config.get("architectures")
    if isinstance(architectures, str) and architectures:
        model["architectures"] = architectures
    elif isinstance(architectures, list) and architectures and all(isinstance(value, str) for value in architectures):
        model["architectures"] = architectures
    elif "architectures" in config:
        raise ValueError("config field architectures must be a non-empty string or list of strings")
    if "architectures" in config:
        evidence.append({
            "claim": "architectures",
            "classification": "config-confirmed",
            "source": "runtime-config",
            "version_or_revision": str(source_revision or "config snapshot; code revision unknown"),
            "locator": f"{config_path}#/architectures",
        })
    resolved_checkpoint = checkpoint or config.get("_name_or_path")
    if isinstance(resolved_checkpoint, str) and resolved_checkpoint:
        model["checkpoint"] = resolved_checkpoint
    model["name"] = str(resolved_checkpoint or model_type or model["name"])
    if source_revision:
        implementation["revision"] = source_revision

    fields = {
        "num_hidden_layers": ("num_hidden_layers", "n_layer", "num_layers"),
        "hidden_size": ("hidden_size", "n_embd", "d_model"),
        "num_attention_heads": ("num_attention_heads", "n_head", "num_heads"),
        "num_key_value_heads": ("num_key_value_heads", "num_kv_heads"),
        "num_experts": ("n_routed_experts", "num_local_experts", "num_experts"),
        "num_experts_per_tok": ("num_experts_per_tok", "num_experts_per_token", "num_selected_experts"),
    }
    for field, aliases in fields.items():
        structure[field] = observed(field, aliases)

    query_heads = structure["num_attention_heads"]
    kv_heads = structure["num_key_value_heads"]
    if isinstance(query_heads, int) and isinstance(kv_heads, int):
        layout = "MHA" if query_heads == kv_heads else "GQA" if query_heads > kv_heads else "unknown"
        if layout != "unknown":
            structure["attention_layout"] = layout
            if layout.lower() not in structure["features"]:
                structure["features"].append(layout.lower())
            evidence.append({
                "claim": "attention_layout",
                "classification": "inferred",
                "source": "runtime-config",
                "version_or_revision": str(source_revision or "config snapshot; code revision unknown"),
                "locator": f"{config_path}#/{nested_prefix}{fields['num_attention_heads'][0]} and num_key_value_heads",
            })
    experts = structure["num_experts"]
    if isinstance(experts, int):
        top_k = structure["num_experts_per_tok"]
        structure["moe_layout"] = f"MoE: {experts} experts; top_k={top_k}" if isinstance(top_k, int) else f"MoE: {experts} experts; top_k=unknown"
        if "moe" not in structure["features"]:
            structure["features"].append("moe")
            evidence.append({
                "claim": "moe",
                "classification": "inferred",
                "source": "runtime-config",
                "version_or_revision": str(source_revision or "config snapshot; code revision unknown"),
                "locator": f"{config_path}#/{nested_prefix}{fields['num_experts'][0]}",
            })

    tied = effective.get("tie_word_embeddings", effective.get("tie_embeddings"))
    if isinstance(tied, bool):
        structure["parameter_sharing"] = f"tie_word_embeddings={str(tied).lower()}"
        evidence.append({
            "claim": "parameter_sharing",
            "classification": "config-confirmed",
            "source": "runtime-config",
            "version_or_revision": str(source_revision or "config snapshot; code revision unknown"),
            "locator": f"{config_path}#/{nested_prefix}{'tie_word_embeddings' if 'tie_word_embeddings' in effective else 'tie_embeddings'}",
        })

    special_keys = ("sliding_window", "rope_scaling", "q_lora_rank", "kv_lora_rank", "use_cache", "ssm_cfg")
    for key in special_keys:
        if key in effective:
            structure["special_paths"].append(f"config.{key}={json.dumps(effective[key], ensure_ascii=False, sort_keys=True)}")
            evidence.append({
                "claim": f"special_path:{key}",
                "classification": "config-confirmed",
                "source": "runtime-config",
                "version_or_revision": str(source_revision or "config snapshot; code revision unknown"),
                "locator": f"{config_path}#/{nested_prefix}{key}",
            })

    vision_config = config.get("vision_config")
    if isinstance(vision_config, dict):
        for key in ("depth", "embed_dim", "num_heads", "patch_size", "spatial_merge_size", "temporal_patch_size"):
            if key not in vision_config:
                continue
            value = vision_config[key]
            structure["special_paths"].append(f"config.vision_config.{key}={json.dumps(value, ensure_ascii=False, sort_keys=True)}")
            evidence.append({
                "claim": f"vision_config:{key}",
                "classification": "config-confirmed",
                "source": "runtime-config",
                "version_or_revision": str(source_revision or "config snapshot; code revision unknown"),
                "locator": f"{config_path}#/vision_config/{key}",
            })

    result["required_runtime_confirmation"] = [
        "核对 config 中的 model_type/architectures 与实际加载类及 state_dict key",
        "锁定源码仓库 commit/tag；传入 --source-revision 记录版本",
        "从源码和 reference forward 补全模块路径、依赖、融合关系及 tensor shape/dtype/layout",
        "验证配置推导出的 Attention/MoE 布局与实际计算图一致",
    ]
    result["checklist"] = make_checklist(structure["features"])
    return result


def load_state_dict_keys(path: Path) -> list[str]:
    """Load a lightweight state-dict key manifest, without importing torch.

    Accepted inputs are a JSON list of keys, a Hugging Face index JSON with a
    ``weight_map`` object, or a newline-delimited text file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read state-dict key manifest: {exc}") from exc
    keys, _ = _parse_state_dict_manifest(text)
    return keys


def _parse_state_dict_manifest(text: str) -> tuple[list[str], bool]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        keys = [line.strip() for line in text.splitlines() if line.strip()]
        complete = False
    else:
        if isinstance(payload, list):
            keys = payload
            complete = False
        elif isinstance(payload, dict) and isinstance(payload.get("weight_map"), dict):
            keys = list(payload["weight_map"])
            complete = payload.get("complete", True)
        elif isinstance(payload, dict) and isinstance(payload.get("keys"), list):
            keys = payload["keys"]
            complete = payload.get("complete", False)
        else:
            raise ValueError("state-dict manifest must be a JSON key list, keys object, weight_map object, or newline text")
    if not isinstance(complete, bool):
        raise ValueError("state-dict manifest field complete must be a boolean")
    if not keys or not all(isinstance(key, str) and key for key in keys):
        raise ValueError("state-dict manifest must contain non-empty string keys")
    return sorted(set(keys)), complete


def load_state_dict_manifest(path: Path) -> tuple[list[str], bool]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read state-dict key manifest: {exc}") from exc
    return _parse_state_dict_manifest(text)


def apply_state_dict_keys(
    record: dict[str, object],
    keys: list[str],
    manifest_path: str,
    complete: bool = False,
) -> dict[str, object]:
    """Overlay module evidence from state-dict names.

    Names reveal module presence and layer/expert indices, but do not prove
    tensor shapes, execution order, fusion, or numerical behavior.
    """
    if not keys or not all(isinstance(key, str) and key for key in keys):
        raise ValueError("state-dict keys must be a non-empty list of strings")
    result = copy.deepcopy(record)
    structure = result["structure"]
    evidence = result["evidence"]
    implementation = result["implementation"]
    key_text = "\n".join(keys)

    def add_evidence(claim: str, locator: str) -> None:
        evidence.append({
            "claim": claim,
            "classification": "state-dict-confirmed",
            "source": "runtime-state-dict",
            "version_or_revision": implementation["revision"],
            "locator": f"{manifest_path}#{locator}",
        })

    layer_matches = re.findall(r"(?:^|\.)(?:layers|h|blocks|transformer_blocks|resblocks)\.(\d+)(?:\.|$)", key_text)
    if layer_matches:
        layer_ids = sorted({int(value) for value in layer_matches})
        if layer_ids == list(range(layer_ids[-1] + 1)):
            observed_layers = len(layer_ids)
            current = structure["num_hidden_layers"]
            if isinstance(current, int):
                if complete and observed_layers != current:
                    raise ValueError(f"complete state-dict layer count {observed_layers} conflicts with config/profile layer count {current}")
                if not complete and observed_layers > current:
                    raise ValueError(f"state-dict layer index {layer_ids[-1]} exceeds config/profile layer count {current}")
                add_evidence("num_hidden_layers" if complete else "layer_index_coverage", f"keys with layer indices 0..{layer_ids[-1]}; {'complete manifest' if complete else 'total count not inferred from partial manifest'}")
            elif complete:
                structure["num_hidden_layers"] = observed_layers
                add_evidence("num_hidden_layers", f"keys with layer indices 0..{layer_ids[-1]}; manifest treated as complete because no config value is available")
            else:
                add_evidence("layer_index_coverage", f"keys with layer indices 0..{layer_ids[-1]}; manifest completeness is unknown")

    expert_matches = re.findall(r"(?:^|\.)experts\.(\d+)(?:\.|$)", key_text)
    if expert_matches:
        expert_ids = sorted({int(value) for value in expert_matches})
        if expert_ids == list(range(expert_ids[-1] + 1)):
            observed_experts = len(expert_ids)
            current = structure["num_experts"]
            if isinstance(current, int):
                if complete and observed_experts != current:
                    raise ValueError(f"complete state-dict expert count {observed_experts} conflicts with config/profile expert count {current}")
                if not complete and observed_experts > current:
                    raise ValueError(f"state-dict expert index {expert_ids[-1]} exceeds config/profile expert count {current}")
                add_evidence("num_experts" if complete else "expert_index_coverage", f"keys with expert indices 0..{expert_ids[-1]}; {'complete manifest' if complete else 'total count not inferred from partial manifest'}")
            elif complete:
                structure["num_experts"] = observed_experts
                structure["moe_layout"] = f"MoE: {observed_experts} experts; top_k={structure['num_experts_per_tok']}"
                if "moe" not in structure["features"]:
                    structure["features"].append("moe")
                add_evidence("num_experts", f"keys with expert indices 0..{expert_ids[-1]}; manifest treated as complete because no config value is available")
            else:
                add_evidence("expert_index_coverage", f"keys with expert indices 0..{expert_ids[-1]}; manifest completeness is unknown")

    patterns = (
        ("attention", r"(?:q_proj|k_proj|v_proj|qkv|out_proj|o_proj)"),
        ("mlp", r"(?:mlp|feed_forward|ffn)"),
        ("router", r"(?:router|block_sparse_moe|moe_gate)"),
        ("vision", r"(?:vision|visual|image|patch_embed)"),
        ("connector", r"(?:connector|merger|multi_modal_projector)"),
        ("timestep", r"(?:time_embed|timestep|ada(?:ln|norm))"),
        ("state", r"(?:state|conv1d|in_proj)"),
        ("embedding", r"(?:embed_tokens|embedding|wte)"),
        ("lm_head", r"(?:lm_head|output)"),
    )
    dependencies = structure["module_dependencies"]
    if isinstance(dependencies, str):
        dependencies = [dependencies]
    for label, pattern in patterns:
        match = next((key for key in keys if re.search(pattern, key, re.IGNORECASE)), None)
        if match is None:
            continue
        marker = f"state-dict:{label}"
        if marker not in structure["special_paths"]:
            structure["special_paths"].append(marker)
        if label not in dependencies:
            dependencies.append(label)
        add_evidence(f"module:{label}", f"key={match}")
    structure["module_dependencies"] = dependencies
    if "router" in dependencies and "moe" not in structure["features"]:
        structure["features"].append("moe")
        add_evidence("moe", "router key present")
    result["required_runtime_confirmation"] = [
        "将 state_dict key 与实际加载类和源码模块路径交叉核对",
        "state_dict 只能确认参数命名和部分模块存在，不能确认执行顺序、融合关系、tensor shape 或数值行为",
        "继续用 reference forward 记录真实 tensor shape/dtype/layout，并验证配置推导出的结构",
    ]
    result["checklist"] = make_checklist(structure["features"])
    return result


def load_forward_manifest(path: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Load reference-forward observations and their run provenance."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read forward manifest JSON: {exc}") from exc
    observations = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(observations, list) or not observations:
        raise ValueError("forward manifest must contain a non-empty observations list")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        raise ValueError("forward manifest metadata must be an object")
    for field in ("source_revision", "device", "dtype", "input_signature"):
        if field in metadata and (not isinstance(metadata[field], str) or not metadata[field]):
            raise ValueError(f"forward manifest metadata.{field} must be a non-empty string")
    for index, item in enumerate(observations):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError(f"forward observation {index} must have a non-empty name")
        if "shape" in item and (not isinstance(item["shape"], list) or not all(isinstance(value, int) for value in item["shape"])):
            raise ValueError(f"forward observation {index} shape must be a list of integers")
        if "dtype" in item and (not isinstance(item["dtype"], str) or not item["dtype"]):
            raise ValueError(f"forward observation {index} dtype must be a non-empty string")
        if "layout" in item and (not isinstance(item["layout"], str) or not item["layout"]):
            raise ValueError(f"forward observation {index} layout must be a non-empty string")
    return observations, {field: metadata[field] for field in ("source_revision", "device", "dtype", "input_signature") if field in metadata}


def apply_forward_manifest(
    record: dict[str, object],
    observations: list[dict[str, object]],
    manifest_path: str,
    metadata: dict[str, str] | None = None,
) -> dict[str, object]:
    """Record shape/dtype/layout observations from a reference forward run."""
    if not observations:
        raise ValueError("forward observations must be non-empty")
    result = copy.deepcopy(record)
    structure = result["structure"]
    evidence = result["evidence"]
    implementation = result["implementation"]
    runtime = result["runtime"]
    metadata = metadata or {}
    for field, value in metadata.items():
        runtime[field] = value
        evidence.append({
            "claim": f"forward_metadata:{field}",
            "classification": "forward-confirmed",
            "source": "runtime-forward",
            "version_or_revision": implementation["revision"],
            "locator": f"{manifest_path}#/metadata/{field}",
        })
    summaries = []
    for index, item in enumerate(observations):
        name = item["name"]
        shape = item.get("shape", "unknown")
        dtype = item.get("dtype", "unknown")
        layout = item.get("layout", "unknown")
        summaries.append(f"{name}: shape={shape}, dtype={dtype}, layout={layout}")
        evidence.append({
            "claim": f"tensor_flow:{name}",
            "classification": "forward-confirmed",
            "source": "runtime-forward",
            "version_or_revision": implementation["revision"],
            "locator": f"{manifest_path}#/observations/{index}",
        })
    structure["tensor_flow"] = "; ".join(summaries)
    result["required_runtime_confirmation"] = [
        "保存 reference forward 的输入、模型代码 revision、dtype、设备和运行配置",
        "对关键节点核对 shape/dtype/layout 与 state_dict、源码模块路径和配置字段",
        "仅凭一次 forward 不能证明所有序列长度、模态路径、timestep、cache 或 sampler 场景",
    ]
    return result


def validate_preflight(record: dict[str, object]) -> list[str]:
    errors = []
    required = {"schema_version", "profile_id", "knowledge_page", "model", "implementation", "runtime", "structure", "evidence", "checklist", "required_runtime_confirmation"}
    if set(record) != required:
        errors.append("record fields do not match the preflight contract")
    if record.get("schema_version") != 1:
        errors.append("unsupported preflight schema_version")
    for field in ("profile_id", "knowledge_page"):
        if not isinstance(record.get(field), str) or not record[field]:
            errors.append(f"{field} must be a non-empty string")
    model = record.get("model", {})
    if not isinstance(model, dict) or not all(isinstance(model.get(key), str) and model[key] for key in ("name", "checkpoint", "model_type")):
        errors.append("model identity fields must be non-empty strings, using 'unknown' when unresolved")
    elif not (isinstance(model.get("architectures"), str) and model["architectures"] or isinstance(model.get("architectures"), list) and model["architectures"] and all(isinstance(value, str) for value in model["architectures"])):
        errors.append("architectures must be a non-empty string or list of strings")
    implementation = record.get("implementation", {})
    if not isinstance(implementation, dict) or set(implementation) != {"repository", "revision"}:
        errors.append("implementation must include repository and revision")
    elif urlparse(implementation["repository"]).scheme not in {"http", "https"} or not implementation["revision"]:
        errors.append("implementation repository/revision is invalid")
    runtime = record.get("runtime", {})
    runtime_fields = {"source_revision", "device", "dtype", "input_signature"}
    if not isinstance(runtime, dict) or set(runtime) != runtime_fields or not all(isinstance(runtime.get(field), str) and runtime[field] for field in runtime_fields):
        errors.append("runtime must include non-empty source_revision, device, dtype, and input_signature")
    structure = record.get("structure", {})
    expected_structure = {"features", "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "num_experts", "num_experts_per_tok", "attention_layout", "moe_layout", "module_dependencies", "parameter_sharing", "special_paths", "tensor_flow"}
    if not isinstance(structure, dict) or set(structure) != expected_structure:
        errors.append("structure fields do not match the preflight contract")
    for item in record.get("evidence", []):
        if not isinstance(item, dict) or set(item) != {"claim", "classification", "source", "version_or_revision", "locator"}:
            errors.append("evidence entry is incomplete")
            continue
        if item["classification"] not in {"source-reported", "source-confirmed", "config-confirmed", "state-dict-confirmed", "forward-confirmed", "inferred", "unknown"}:
            errors.append(f"invalid evidence classification: {item['classification']}")
    for item in record.get("checklist", []):
        if not isinstance(item, dict) or set(item) != {"feature", "pages", "potential_risk", "check_method", "validation_status"}:
            errors.append("checklist entry is incomplete")
            continue
        if item["validation_status"] != "待核实":
            errors.append("unvalidated structure items must remain marked 待核实")
        pages = item["pages"]
        if not isinstance(pages, list) or not pages or not all(isinstance(page, str) and page for page in pages):
            errors.append("checklist pages must be a non-empty list of page IDs")
        else:
            try:
                known = {page.meta["id"] for page in pages_of(ROOT)}
            except ValueError as exc:
                errors.append(f"cannot load pages: {exc}")
            else:
                unknown = [page for page in pages if page not in known]
                if unknown:
                    errors.append(f"checklist references unregistered pages: {unknown}")
    if not isinstance(record.get("required_runtime_confirmation"), list):
        errors.append("required_runtime_confirmation must be a list")
    return errors


def runtime_gaps(record: dict[str, object]) -> list[str]:
    """Return evidence gaps that block structure-dependent quantization."""
    gaps = []
    model = record.get("model", {})
    implementation = record.get("implementation", {})
    runtime = record.get("runtime", {})
    structure = record.get("structure", {})
    evidence = record.get("evidence", [])
    if model.get("checkpoint") == "unknown":
        gaps.append("checkpoint 未锁定")
    has_signature = any(item.get("claim") == "architecture_signature" for item in evidence if isinstance(item, dict))
    if model.get("model_type") == "unknown" and not has_signature:
        gaps.append("model_type 未从运行时配置确认")
    if model.get("architectures") == "unknown":
        gaps.append("architectures 未从运行时配置确认")
    if implementation.get("revision") == "unknown":
        gaps.append("源码 repository revision 未锁定")
    for field in ("source_revision", "device", "dtype", "input_signature"):
        if runtime.get(field) == "unknown":
            gaps.append(f"runtime.{field} 未记录")
    for field in ("num_hidden_layers", "hidden_size", "num_attention_heads"):
        if structure.get(field) == "unknown":
            gaps.append(f"structure.{field} 未确认")
    features = set(structure.get("features", []))
    if features & {"attention", "mha", "gqa", "mla"}:
        if structure.get("attention_layout") not in {"MHA", "GQA", "MLA"}:
            gaps.append("Attention 布局未从配置/运行图解析")
    if "gqa" in features and structure.get("num_key_value_heads") == "unknown":
        gaps.append("GQA 的 num_key_value_heads 未确认")
    if "moe" in features:
        for field in ("num_experts", "num_experts_per_tok"):
            if structure.get(field) == "unknown":
                gaps.append(f"MoE 的 structure.{field} 未确认")
    tensor_flow = structure.get("tensor_flow", "")
    if not tensor_flow or tensor_flow.startswith("需从目标 checkpoint"):
        gaps.append("reference forward 的 tensor shape/dtype/layout 未记录")
    classifications = {item.get("classification") for item in evidence if isinstance(item, dict)}
    if "state-dict-confirmed" not in classifications:
        gaps.append("缺少 state_dict key 证据")
    if "forward-confirmed" not in classifications:
        gaps.append("缺少 reference forward 证据")
    forward_claims = {item.get("claim") for item in evidence if isinstance(item, dict) and item.get("classification") == "forward-confirmed"}
    for field in ("source_revision", "device", "dtype", "input_signature"):
        if f"forward_metadata:{field}" not in forward_claims:
            gaps.append(f"reference forward 元数据 {field} 未记录")
    return gaps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_id", nargs="?")
    parser.add_argument("--list", action="store_true", help="list available representative profiles")
    parser.add_argument("--config", type=Path, help="local Hugging Face config.json to inspect")
    parser.add_argument("--checkpoint", help="checkpoint name or path to record")
    parser.add_argument("--source-revision", help="code commit or tag corresponding to the config")
    parser.add_argument("--state-dict-keys", type=Path, help="JSON key list, HF index JSON, or newline-delimited state-dict key manifest")
    parser.add_argument("--forward-manifest", type=Path, help="JSON reference-forward observation manifest with an observations list")
    parser.add_argument("--strict-runtime", action="store_true", help="fail unless runtime structure evidence is complete enough for quantization")
    parser.add_argument("--quantization-view", action="store_true",
                        help="emit the quantization-stage view for an Agent; not necessarily smaller than the default record")
    args = parser.parse_args()
    profiles = load_profiles()
    if args.list:
        for profile in profiles:
            print(f"{profile['id']}\t{profile['model_name']}\t{profile['knowledge_page']}")
        return 0
    config = None
    if args.config:
        try:
            config = json.loads(args.config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot read config JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(config, dict):
            print("model config JSON root must be an object", file=sys.stderr)
            return 2
    try:
        profile = select_profile(profiles, args.profile_id, config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    result = build_preflight(profile)
    if args.config:
        try:
            result = apply_runtime_config(result, config, str(args.config), args.checkpoint, args.source_revision, profile.get("model_type_hints", []), profile.get("config_signatures", []))
        except ValueError as exc:
            print(f"cannot apply config to profile: {exc}", file=sys.stderr)
            return 2
    if args.state_dict_keys:
        try:
            keys, complete = load_state_dict_manifest(args.state_dict_keys)
            result = apply_state_dict_keys(result, keys, str(args.state_dict_keys), complete=complete)
        except ValueError as exc:
            print(f"cannot apply state-dict keys: {exc}", file=sys.stderr)
            return 2
    if args.forward_manifest:
        try:
            observations, metadata = load_forward_manifest(args.forward_manifest)
            result = apply_forward_manifest(result, observations, str(args.forward_manifest), metadata)
        except ValueError as exc:
            print(f"cannot apply forward manifest: {exc}", file=sys.stderr)
            return 2
    errors = validate_preflight(result)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if args.strict_runtime:
        gaps = runtime_gaps(result)
        if gaps:
            print("strict runtime preflight blocked:", file=sys.stderr)
            print("\n".join(f"- {gap}" for gap in gaps), file=sys.stderr)
            return 3
    output = quantization_view(result) if args.quantization_view else result
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
