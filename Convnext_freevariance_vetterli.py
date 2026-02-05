"""
MonoSoup script for ConvNeXt Small (timm).

This script:
1. Builds a paired ConvNeXt setup:
   - W0: convnext_small.in12k body + IN1k classifier head
   - W1: convnext_small.in12k_ft_in1k full model
2. Applies layer-wise MonoSoup between W0 and W1.
3. Optionally evaluates on ImageNet + OOD datasets from model-soups.
4. Exports layer-level lambda statistics and run metadata.

Example:
python Convnext_freevariance_vetterli.py \
  --data-location /path/to/datasets_root \
  --version vetterli \
  --R 0.8 \
  --device auto \
  --output-json convnext_run.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
import re
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn


LOGGER = logging.getLogger("convnext_monosoup")

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_SOUPS_ROOT = ROOT_DIR / "model-soups"
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "ConvNext"

DATASET_CLASS_NAMES = (
    "ImageNet",
    "ImageNetV2",
    "ImageNetSketch",
    "ImageNetR",
    "ObjectNet",
    "ImageNetA",
)


@dataclass(frozen=True)
class RunConfig:
    data_location: str = str(ROOT_DIR.resolve())
    model_soups_root: Path = DEFAULT_MODEL_SOUPS_ROOT
    device: str = "auto"
    version: str = "vetterli"
    r_threshold: float = 0.9
    min_rel_update: float = 1e-2
    batch_size: int = 256
    seed: int = 0
    debug_differences: bool = False
    verbose_layers: bool = False
    skip_eval: bool = False
    save_layer_stats: bool = True
    layer_stats_path: Path | None = None
    output_json: Path | None = None
    in12k_model_name: str = "convnext_small.in12k"
    in1k_model_name: str = "convnext_small.in12k_ft_in1k"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_version(version: str) -> str:
    if version == "freevariance":
        return "vetterli"
    return version


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(device_arg)


def import_model_soups_datasets(model_soups_root: Path):
    """
    Import dataset classes from the local model-soups repository.
    """
    model_soups_root = model_soups_root.resolve()
    if not model_soups_root.exists():
        raise FileNotFoundError(f"model-soups root not found: {model_soups_root}")

    root_str = str(model_soups_root)
    if root_str not in sys.path:
        # Keep local model-soups first to avoid huggingface datasets module shadowing.
        sys.path.insert(0, root_str)

    datasets_module = importlib.import_module("datasets")
    missing = [name for name in DATASET_CLASS_NAMES if not hasattr(datasets_module, name)]
    if missing:
        raise ImportError(
            "Missing required dataset classes in model-soups datasets.py: "
            f"{missing}. Imported module: {datasets_module.__file__}"
        )
    return [getattr(datasets_module, name) for name in DATASET_CLASS_NAMES]


def maybe_dictionarize_batch(batch):
    if isinstance(batch, dict):
        return batch
    if len(batch) == 2:
        return {"images": batch[0], "labels": batch[1]}
    if len(batch) == 3:
        return {"images": batch[0], "labels": batch[1], "metadata": batch[2]}
    raise ValueError(f"Unexpected number of elements in batch: {len(batch)}")


@torch.inference_mode()
def test_model_on_dataset(model: nn.Module, dataset, device: torch.device) -> float:
    """
    Evaluate a model on one model-soups dataset object.
    """
    model.eval()
    correct, n = 0.0, 0.0
    end = time.time()

    loader = dataset.test_loader
    if type(dataset).__name__ == "ImageNet2p":
        loader = dataset.train_loader
        assert dataset.train_dataset.__getitem__(dataset.sampler.indices[1000])[
            "image_paths"
        ].endswith("n01675722_4108.JPEG")

    for i, batch in enumerate(loader):
        batch = maybe_dictionarize_batch(batch)
        inputs = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        image_paths = batch.get("image_paths", None)
        data_time = time.time() - end

        logits = model(inputs)
        targets = labels

        projection_fn = getattr(dataset, "project_logits", None)
        if projection_fn is not None:
            logits = projection_fn(logits, device.type)

        if hasattr(dataset, "project_labels"):
            targets = dataset.project_labels(targets, device.type)

        if isinstance(logits, list):
            logits = logits[0]

        pred = logits.argmax(dim=1, keepdim=True)
        if hasattr(dataset, "accuracy"):
            acc1, num_total = dataset.accuracy(logits, targets, image_paths, None)
            correct += acc1
            n += num_total
        else:
            correct += pred.eq(targets.view_as(pred)).sum().item()
            n += targets.size(0)

        batch_time = time.time() - end
        end = time.time()
        if i % 20 == 0:
            pct = 100.0 * i / len(loader)
            LOGGER.info(
                "[%3.0f%% %d/%d] Acc: %.2f | Data %.3fs | Batch %.3fs",
                pct,
                i,
                len(loader),
                100.0 * (correct / max(n, 1)),
                data_time,
                batch_time,
            )

    return float(correct / max(n, 1))


@torch.inference_mode()
def eval_model(
    model: nn.Module,
    preprocess,
    dataset_classes: Sequence[type],
    data_location: str,
    batch_size: int,
    device: torch.device,
    model_name: str = "ConvNeXt MonoSoup",
) -> Dict[str, float]:
    """
    Evaluate model on ImageNet and OOD datasets.
    """
    results: Dict[str, float] = {}
    for dataset_cls in dataset_classes:
        LOGGER.info("[%s] Evaluating on %s", model_name, dataset_cls.__name__)
        dataset = dataset_cls(preprocess, data_location, batch_size)
        acc = test_model_on_dataset(model, dataset, device=device)
        LOGGER.info("[%s] %s accuracy: %.4f", model_name, dataset_cls.__name__, acc)
        results[dataset_cls.__name__] = acc

    id_acc = results["ImageNet"]
    ood_scores = [value for key, value in results.items() if key != "ImageNet"]
    avg_ood = float(np.mean(ood_scores)) if ood_scores else float("nan")
    LOGGER.info("[%s] In-distribution (ImageNet): %.4f", model_name, id_acc)
    LOGGER.info("[%s] Avg OOD: %.4f", model_name, avg_ood)
    return results


def load_pretrained_and_finetuned_convnext(
    device: torch.device,
    in12k_model_name: str,
    in1k_model_name: str,
) -> Tuple[nn.Module, nn.Module, Any]:
    """
    Build paired ConvNeXt models with identical IN1k architecture:
      - W0: IN12k body + IN1k head
      - W1: IN1k fine-tuned model
    """
    LOGGER.info("Loading ConvNeXt models: %s and %s", in12k_model_name, in1k_model_name)

    base_arch = timm.create_model(in1k_model_name, pretrained=False)
    base_arch.eval()

    data_config = timm.data.resolve_model_data_config(base_arch)
    preprocess = timm.data.create_transform(**data_config, is_training=False)

    model_in12k = timm.create_model(in12k_model_name, pretrained=True)
    model_in1k = timm.create_model(in1k_model_name, pretrained=True)

    sd_in12k = model_in12k.state_dict()
    sd_in1k = model_in1k.state_dict()

    pretrained_model = deepcopy(base_arch)
    finetuned_model = deepcopy(base_arch)
    base_sd = base_arch.state_dict()

    pre_sd_aligned = {}
    for name, tensor in base_sd.items():
        if name in sd_in12k and sd_in12k[name].shape == tensor.shape:
            pre_sd_aligned[name] = sd_in12k[name]
        elif name in sd_in1k and sd_in1k[name].shape == tensor.shape:
            pre_sd_aligned[name] = sd_in1k[name]
        else:
            pre_sd_aligned[name] = tensor

    ft_sd_aligned = {}
    for name, tensor in base_sd.items():
        if name in sd_in1k and sd_in1k[name].shape == tensor.shape:
            ft_sd_aligned[name] = sd_in1k[name]
        else:
            ft_sd_aligned[name] = tensor

    pretrained_model.load_state_dict(pre_sd_aligned)
    finetuned_model.load_state_dict(ft_sd_aligned)

    pretrained_model.to(device).eval()
    finetuned_model.to(device).eval()
    LOGGER.info("ConvNeXt model pair is ready.")
    return pretrained_model, finetuned_model, preprocess


def _reshape_to_matrix(weight: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if weight.ndim < 2:
        raise ValueError("Weight tensor must have at least 2 dimensions.")
    original_shape = tuple(weight.shape)
    rows = weight.shape[0]
    cols = int(weight.numel() // rows)
    return weight.reshape(rows, cols), original_shape


def _reshape_from_matrix(matrix: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
    return matrix.reshape(original_shape)


def should_process_param(
    weight_pretrained: torch.Tensor,
    weight_finetuned: torch.Tensor,
    min_rel_update: float = 1e-6,
    eps: float = 1e-12,
) -> bool:
    if weight_finetuned.ndim < 2:
        return False

    delta = (weight_finetuned - weight_pretrained).detach()
    update_norm = torch.linalg.norm(delta)
    base_norm = torch.linalg.norm(weight_pretrained.detach())

    if base_norm < eps and update_norm < eps:
        return False
    if base_norm < eps:
        return True

    rel_update = (update_norm / (base_norm + eps)).item()
    return rel_update >= min_rel_update


def _choose_k_and_pk(
    singular_values: torch.Tensor,
    singular_values_sq: torch.Tensor,
    total_energy: torch.Tensor,
    r_threshold: float,
    version: str,
    eps: float = 1e-12,
) -> Tuple[int, int, torch.Tensor]:
    rank = singular_values.shape[0]

    if version == "vetterli":
        s_abs = singular_values.abs()
        total_mag = s_abs.sum()
        if total_mag < eps:
            num_keep = rank
            k_index = rank - 1
            p_k = torch.tensor(1.0, device=singular_values.device, dtype=singular_values.dtype)
            return num_keep, k_index, p_k

        probs = s_abs / (total_mag + eps)
        entropy = -(probs * torch.log(probs + eps)).sum()
        effective_rank = torch.exp(entropy)
        num_keep = int(torch.clamp(torch.ceil(effective_rank), 1, rank).item())
        k_index = num_keep - 1
        p_k = singular_values_sq[:num_keep].sum() / (total_energy + eps)
        return num_keep, k_index, p_k

    if version == "variance":
        cumulative_energy = singular_values_sq.cumsum(0) / (total_energy + eps)
        k_index = int(torch.searchsorted(cumulative_energy, r_threshold, right=False).item())
        k_index = max(0, min(k_index, rank - 1))
        num_keep = k_index + 1
        p_k = cumulative_energy[k_index]
        return num_keep, k_index, p_k

    raise ValueError(f"Unknown version '{version}'. Use 'variance' or 'vetterli'.")


def _monosoup_update_for_layer(
    layer_name: str,
    weight_pretrained: torch.Tensor,
    weight_finetuned: torch.Tensor,
    r_threshold: float,
    version: str,
    eps: float = 1e-12,
    verbose: bool = False,
    return_stats: bool = False,
) -> Any:
    """
    Apply MonoSoup to one weight tensor.
    """
    if weight_pretrained.shape != weight_finetuned.shape:
        raise ValueError(
            f"Shape mismatch for layer {layer_name}: "
            f"{weight_pretrained.shape} vs {weight_finetuned.shape}"
        )

    device = weight_finetuned.device
    output_dtype = weight_finetuned.dtype

    w0_matrix, original_shape = _reshape_to_matrix(weight_pretrained)
    w1_matrix, _ = _reshape_to_matrix(weight_finetuned)
    delta = w1_matrix - w0_matrix

    svd_dtype = torch.float32 if delta.dtype in (torch.float16, torch.bfloat16) else delta.dtype
    delta_svd = delta.to(svd_dtype)
    u, singular_values, v_h = torch.linalg.svd(delta_svd, full_matrices=False)

    singular_values_sq = singular_values.square()
    total_energy = singular_values_sq.sum()
    if total_energy < eps:
        if verbose:
            LOGGER.info("[MonoSoup] Layer '%s': spectral energy ~ 0, copied from W1.", layer_name)
        edited_weight = weight_finetuned.clone()
        if not return_stats:
            return edited_weight

        return edited_weight, {
            "name": layer_name,
            "num_keep": 0,
            "k_index": -1,
            "P_k": 0.0,
            "rho": 0.0,
            "cos_alpha": 0.0,
            "lambda_high": 0.0,
            "lambda_low": 1.0,
            "num_singular": int(singular_values.shape[0]),
            "matrix_rows": int(w0_matrix.shape[0]),
            "matrix_cols": int(w0_matrix.shape[1]),
            "spectral_zero": True,
        }

    num_keep, k_index, p_k = _choose_k_and_pk(
        singular_values=singular_values,
        singular_values_sq=singular_values_sq,
        total_energy=total_energy,
        r_threshold=r_threshold,
        version=version,
        eps=eps,
    )

    u_k = u[:, :num_keep]
    s_k = singular_values[:num_keep]
    v_h_k = v_h[:num_keep, :]
    delta_high = (u_k * s_k) @ v_h_k
    delta_low = delta_svd - delta_high

    if num_keep < singular_values.shape[0]:
        sigma_kplus1 = singular_values[num_keep]
        sigma_1 = singular_values[0]
        rho = (sigma_kplus1 / (sigma_1 + eps)).square()
    else:
        rho = torch.tensor(0.0, dtype=delta_svd.dtype, device=delta_svd.device)

    cos_alpha = torch.sqrt(torch.clamp(1.0 - p_k, min=0.0, max=1.0))
    lambda_low = rho + (1.0 - rho) * cos_alpha
    lambda_high = 1.0 - lambda_low

    lambda_low = torch.clamp(lambda_low, 0.0, 1.0)
    lambda_high = torch.clamp(lambda_high, 0.0, 1.0)

    if verbose:
        LOGGER.info(
            "[MonoSoup] %-70s num_keep=%-4d P_k=%.4f rho=%.3e lambda_high=%.4f lambda_low=%.4f",
            layer_name,
            num_keep,
            p_k.item(),
            rho.item(),
            lambda_high.item(),
            lambda_low.item(),
        )

    delta_mono = lambda_high * delta_high + lambda_low * delta_low
    edited_matrix = w0_matrix.to(svd_dtype) + delta_mono
    edited_weight = _reshape_from_matrix(edited_matrix, original_shape)
    edited_weight = edited_weight.to(device=device, dtype=output_dtype)

    if not return_stats:
        return edited_weight

    return edited_weight, {
        "name": layer_name,
        "num_keep": int(num_keep),
        "k_index": int(k_index),
        "P_k": float(p_k.item()),
        "rho": float(rho.item()),
        "cos_alpha": float(cos_alpha.item()),
        "lambda_high": float(lambda_high.item()),
        "lambda_low": float(lambda_low.item()),
        "num_singular": int(singular_values.shape[0]),
        "matrix_rows": int(w0_matrix.shape[0]),
        "matrix_cols": int(w0_matrix.shape[1]),
        "spectral_zero": False,
    }


def _layer_descriptor(layer_name: str) -> Dict[str, str]:
    """
    Create a short readable descriptor from a ConvNeXt parameter name.
    """
    group = "other"
    block_match = re.search(r"stages\.(\d+)\.blocks\.(\d+)", layer_name)
    if block_match:
        stage = int(block_match.group(1))
        block = int(block_match.group(2))
        group = f"stage{stage}_block{block}"
    else:
        downsample_match = re.search(r"downsample_layers\.(\d+)", layer_name)
        if downsample_match:
            idx = int(downsample_match.group(1))
            group = "stem" if idx == 0 else f"downsample{idx}"
        elif layer_name.startswith("stem"):
            group = "stem"
        elif layer_name.startswith("head") or layer_name.startswith("classifier"):
            group = "head"

    layer_type = "Other"
    if ".mlp." in layer_name:
        if ".fc1." in layer_name:
            layer_type = "MLP/FC1"
        elif ".fc2." in layer_name:
            layer_type = "MLP/FC2"
        else:
            layer_type = "MLP"
    elif ".pwconv1." in layer_name:
        layer_type = "PointwiseConv1"
    elif ".pwconv2." in layer_name:
        layer_type = "PointwiseConv2"
    elif ".conv_dw." in layer_name or ".dwconv." in layer_name:
        layer_type = "DepthwiseConv"
    elif ".norm" in layer_name:
        layer_type = "LayerNorm"
    elif "downsample_layers" in layer_name:
        layer_type = "Downsample"
    elif layer_name.startswith("head") or layer_name.startswith("classifier"):
        layer_type = "Head"

    return {
        "group": group,
        "type": layer_type,
        "label": f"{group} {layer_type}",
    }


def apply_monosoup(
    pretrained: nn.Module,
    finetuned: nn.Module,
    r_threshold: float = 0.8,
    min_rel_update: float = 1e-6,
    version: str = "variance",
    verbose: bool = False,
    stats: list[Dict[str, Any]] | None = None,
) -> nn.Module:
    """
    Apply MonoSoup layer-wise to all matching weight tensors (ndim >= 2).
    """
    if version == "variance" and not (0.0 < r_threshold <= 1.0):
        raise ValueError("r_threshold must be in (0, 1] when version='variance'.")

    edited = deepcopy(finetuned)
    pretrained_params = {
        name: param.detach().clone()
        for name, param in pretrained.named_parameters()
    }

    processed = 0
    skipped_not_found = 0
    skipped_shape_mismatch = 0
    skipped_low_update = 0
    layer_index = 1

    with torch.no_grad():
        for name, p_edit in edited.named_parameters():
            if name not in pretrained_params:
                skipped_not_found += 1
                continue

            p_pre = pretrained_params[name]
            if p_pre.shape != p_edit.shape:
                skipped_shape_mismatch += 1
                continue

            if not should_process_param(
                weight_pretrained=p_pre,
                weight_finetuned=p_edit,
                min_rel_update=min_rel_update,
            ):
                skipped_low_update += 1
                continue

            if stats is None:
                new_weight = _monosoup_update_for_layer(
                    layer_name=name,
                    weight_pretrained=p_pre,
                    weight_finetuned=p_edit,
                    r_threshold=r_threshold,
                    version=version,
                    verbose=verbose,
                    return_stats=False,
                )
            else:
                new_weight, layer_stats = _monosoup_update_for_layer(
                    layer_name=name,
                    weight_pretrained=p_pre,
                    weight_finetuned=p_edit,
                    r_threshold=r_threshold,
                    version=version,
                    verbose=verbose,
                    return_stats=True,
                )
                desc = _layer_descriptor(name)
                layer_stats["index"] = int(layer_index)
                layer_stats["group"] = desc["group"]
                layer_stats["type"] = desc["type"]
                layer_stats["label"] = desc["label"]
                stats.append(layer_stats)
                layer_index += 1

            p_edit.copy_(new_weight)
            processed += 1

    LOGGER.info(
        "MonoSoup summary | processed=%d | skipped_not_found=%d | skipped_shape_mismatch=%d | skipped_low_update=%d",
        processed,
        skipped_not_found,
        skipped_shape_mismatch,
        skipped_low_update,
    )
    return edited


def debug_param_differences(
    pretrained_model: nn.Module,
    finetuned_model: nn.Module,
    threshold: float = 1e-6,
    max_print: int = 20,
) -> None:
    LOGGER.info("[DEBUG] Checking parameter differences between paired checkpoints.")
    diff_count = 0
    ft_params = dict(finetuned_model.named_parameters())

    for name, p_pre in pretrained_model.named_parameters():
        if name not in ft_params:
            continue
        p_ft = ft_params[name]
        if p_pre.shape != p_ft.shape:
            continue
        diff_norm = (p_ft.detach() - p_pre.detach()).norm().item()
        if diff_norm > threshold:
            diff_count += 1
            if diff_count <= max_print:
                LOGGER.info("[DIFF] %s: ||ΔW||_F = %.3e", name, diff_norm)

    LOGGER.info(
        "[DEBUG] Number of params with ||ΔW||_F > %.1e: %d",
        threshold,
        diff_count,
    )


def serialize_config(config: RunConfig) -> Dict[str, Any]:
    config_dict = asdict(config)
    for key in ("model_soups_root", "layer_stats_path", "output_json"):
        if config_dict[key] is not None:
            config_dict[key] = str(config_dict[key])
    return config_dict


def default_layer_stats_path(config: RunConfig) -> Path:
    DEFAULT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    version = normalize_version(config.version)
    return DEFAULT_ARTIFACT_DIR / f"lambda_stats_convnext_{version}_R{config.r_threshold}.json"


def save_layer_stats(
    path: Path,
    config: RunConfig,
    layer_stats: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "model_in12k": config.in12k_model_name,
            "model_in1k": config.in1k_model_name,
            "version": normalize_version(config.version),
            "R": float(config.r_threshold),
            "min_rel_update": float(config.min_rel_update),
            "seed": int(config.seed),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "num_layers": len(layer_stats),
        },
        "layers": list(layer_stats),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("Saved layer lambda stats to: %s", path)


def save_run_results(
    path: Path,
    config: RunConfig,
    results: Mapping[str, float],
    num_layers: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": serialize_config(config),
        "num_processed_layers": int(num_layers),
        "results": dict(results),
        "metrics": {
            "imagenet": results.get("ImageNet"),
            "avg_ood": float(
                np.mean([v for k, v in results.items() if k != "ImageNet"])
            ) if any(k != "ImageNet" for k in results) else None,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    LOGGER.info("Saved run summary to: %s", path)


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Publication-ready ConvNeXt MonoSoup runner (variance/vetterli).",
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default=str(ROOT_DIR.resolve()),
        help="Dataset root used by model-soups dataset wrappers.",
    )
    parser.add_argument(
        "--model-soups-root",
        type=Path,
        default=DEFAULT_MODEL_SOUPS_ROOT,
        help="Path to local model-soups repository.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="'auto' picks CUDA when available.",
    )
    parser.add_argument(
        "--version",
        choices=("variance", "vetterli", "freevariance"),
        default="vetterli",
        help="'freevariance' is accepted as an alias for 'vetterli'.",
    )
    parser.add_argument(
        "--R",
        type=float,
        default=0.9,
        help="Variance threshold used only when --version variance.",
    )
    parser.add_argument("--min-rel-update", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-differences", action="store_true")
    parser.add_argument("--verbose-layers", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--no-layer-stats",
        action="store_true",
        help="Disable export of layer-wise lambda stats JSON.",
    )
    parser.add_argument(
        "--layer-stats-path",
        type=Path,
        default=None,
        help="Optional output path for layer-wise lambda stats JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for run metadata + evaluation JSON.",
    )
    parser.add_argument(
        "--in12k-model-name",
        type=str,
        default="convnext_small.in12k",
        help="timm model identifier for pretrained source.",
    )
    parser.add_argument(
        "--in1k-model-name",
        type=str,
        default="convnext_small.in12k_ft_in1k",
        help="timm model identifier for fine-tuned source and base architecture.",
    )

    args = parser.parse_args(argv)
    return RunConfig(
        data_location=args.data_location,
        model_soups_root=args.model_soups_root,
        device=args.device,
        version=args.version,
        r_threshold=args.R,
        min_rel_update=args.min_rel_update,
        batch_size=args.batch_size,
        seed=args.seed,
        debug_differences=args.debug_differences,
        verbose_layers=args.verbose_layers,
        skip_eval=args.skip_eval,
        save_layer_stats=not args.no_layer_stats,
        layer_stats_path=args.layer_stats_path,
        output_json=args.output_json,
        in12k_model_name=args.in12k_model_name,
        in1k_model_name=args.in1k_model_name,
    )


def run(config: RunConfig) -> Dict[str, float]:
    set_reproducibility(config.seed)
    device = resolve_device(config.device)
    version = normalize_version(config.version)

    LOGGER.info("Resolved device: %s", device)
    LOGGER.info("Run config: %s", json.dumps(serialize_config(config), indent=2, sort_keys=True))

    pretrained_model, finetuned_model, preprocess = load_pretrained_and_finetuned_convnext(
        device=device,
        in12k_model_name=config.in12k_model_name,
        in1k_model_name=config.in1k_model_name,
    )

    if config.debug_differences:
        debug_param_differences(pretrained_model, finetuned_model, threshold=1e-6, max_print=60)

    layer_stats: list[Dict[str, Any]] = []
    edited_model = apply_monosoup(
        pretrained=pretrained_model,
        finetuned=finetuned_model,
        r_threshold=config.r_threshold,
        min_rel_update=config.min_rel_update,
        version=version,
        verbose=config.verbose_layers,
        stats=layer_stats if config.save_layer_stats else None,
    )

    if config.save_layer_stats:
        layer_stats_path = config.layer_stats_path or default_layer_stats_path(config)
        save_layer_stats(layer_stats_path, config, layer_stats)

    results: Dict[str, float] = {}
    if not config.skip_eval:
        dataset_classes = import_model_soups_datasets(config.model_soups_root)
        results = eval_model(
            edited_model,
            preprocess=preprocess,
            dataset_classes=dataset_classes,
            data_location=config.data_location,
            batch_size=config.batch_size,
            device=device,
            model_name=f"ConvNeXt MonoSoup({version})",
        )
    else:
        LOGGER.info("Skipping evaluation (--skip-eval set).")

    if config.output_json is not None:
        save_run_results(config.output_json, config, results, len(layer_stats))

    return results


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    config = parse_args(argv)
    results = run(config)
    if results:
        LOGGER.info("Final results: %s", json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
