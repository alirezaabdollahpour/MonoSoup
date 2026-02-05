"""
Publication-ready ConvNeXt MonoSoup + CKA experiment.

This script:
1. Builds a paired ConvNeXt setup:
   - W0: convnext_small.in12k body + IN1k classifier head
   - W1: convnext_small.in12k_ft_in1k full model
2. Builds three edited variants from layer-wise SVD decomposition:
   - monosoup: W0 + delta_mono
   - highonly: W0 + delta_high
   - lowonly : W0 + delta_low
3. Computes layer-wise linear CKA(pretrained, model) on selected datasets.
4. Optionally evaluates model accuracies and saves reproducible JSON artifacts.

Example:
python CKA_ConvNext.py \
  --data-location /path/to/datasets_root \
  --version vetterli \
  --R 0.8 \
  --cka-datasets ImageNet ImageNetA \
  --max-cka-batches 25 \
  --device auto
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn


matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger("convnext_cka")

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_SOUPS_ROOT = ROOT_DIR / "model-soups"
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "ConvNext" / "cka"

DATASET_CLASS_NAMES = (
    "ImageNet",
    "ImageNetV2",
    "ImageNetSketch",
    "ImageNetR",
    "ObjectNet",
    "ImageNetA",
)

MODEL_ORDER = (
    "pretrained",
    "finetuned",
    "monosoup",
    "highonly",
    "lowonly",
)

MODEL_LABELS = {
    "pretrained": "Pretrained",
    "finetuned": "Fine-tuned",
    "monosoup": "MonoSoup",
    "highonly": "High-only",
    "lowonly": "Low-only",
}


@dataclass(frozen=True)
class RunConfig:
    data_location: str = str(ROOT_DIR.resolve())
    model_soups_root: Path = DEFAULT_MODEL_SOUPS_ROOT
    device: str = "auto"
    version: str = "vetterli"
    r_threshold: float = 0.8
    min_rel_update: float = 1e-2
    batch_size: int = 256
    seed: int = 0
    debug_differences: bool = False
    verbose_layers: bool = False
    run_eval: bool = False
    eval_models: Tuple[str, ...] = MODEL_ORDER
    eval_datasets: Tuple[str, ...] = DATASET_CLASS_NAMES
    cka_datasets: Tuple[str, ...] = ("ImageNet", "ImageNetA")
    max_cka_batches: int = 25
    no_plots: bool = False
    plot_dir: Path = DEFAULT_ARTIFACT_DIR / "plots"
    cka_json_path: Path | None = None
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


def import_model_soups_datasets(model_soups_root: Path) -> Dict[str, type]:
    """
    Import dataset classes from the local model-soups repository.
    """
    model_soups_root = model_soups_root.resolve()
    if not model_soups_root.exists():
        raise FileNotFoundError(f"model-soups root not found: {model_soups_root}")

    root_str = str(model_soups_root)
    if root_str not in sys.path:
        # Keep local model-soups first to avoid HF datasets package shadowing.
        sys.path.insert(0, root_str)

    datasets_module = importlib.import_module("datasets")
    missing = [name for name in DATASET_CLASS_NAMES if not hasattr(datasets_module, name)]
    if missing:
        raise ImportError(
            "Missing required dataset classes in model-soups datasets.py: "
            f"{missing}. Imported module: {datasets_module.__file__}"
        )

    return {name: getattr(datasets_module, name) for name in DATASET_CLASS_NAMES}


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
    Evaluate a model on one model-soups dataset.
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
    dataset_names: Sequence[str],
    dataset_classes: Mapping[str, type],
    data_location: str,
    batch_size: int,
    device: torch.device,
    model_name: str,
) -> Dict[str, float]:
    """
    Evaluate model on selected datasets and return accuracy by dataset.
    """
    results: Dict[str, float] = {}
    for dataset_name in dataset_names:
        dataset_cls = dataset_classes[dataset_name]
        LOGGER.info("[%s] Evaluating on %s", model_name, dataset_name)
        dataset = dataset_cls(preprocess, data_location, batch_size)
        acc = test_model_on_dataset(model, dataset, device=device)
        LOGGER.info("[%s] %s accuracy: %.4f", model_name, dataset_name, acc)
        results[dataset_name] = acc

    if "ImageNet" in results:
        id_acc = results["ImageNet"]
        ood_scores = [v for k, v in results.items() if k != "ImageNet"]
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


def _monosoup_components_for_layer(
    layer_name: str,
    weight_pretrained: torch.Tensor,
    weight_finetuned: torch.Tensor,
    r_threshold: float,
    version: str,
    eps: float = 1e-12,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute layer-wise edited weights for three variants:
      - MonoSoup   : W0 + delta_mono
      - High-only  : W0 + delta_high
      - Low-only   : W0 + delta_low
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
        base = weight_finetuned.clone()
        return base.clone(), base.clone(), base.clone()

    num_keep, _, p_k = _choose_k_and_pk(
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
    w_mono = w0_matrix.to(svd_dtype) + delta_mono
    w_high = w0_matrix.to(svd_dtype) + delta_high
    w_low = w0_matrix.to(svd_dtype) + delta_low

    mono_weight = _reshape_from_matrix(w_mono, original_shape).to(device=device, dtype=output_dtype)
    high_weight = _reshape_from_matrix(w_high, original_shape).to(device=device, dtype=output_dtype)
    low_weight = _reshape_from_matrix(w_low, original_shape).to(device=device, dtype=output_dtype)
    return mono_weight, high_weight, low_weight


def apply_monosoup_with_ablation(
    pretrained: nn.Module,
    finetuned: nn.Module,
    r_threshold: float = 0.8,
    min_rel_update: float = 1e-6,
    version: str = "variance",
    verbose: bool = False,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Apply MonoSoup and build three edited variants:
      - monosoup model
      - high-only model
      - low-only model
    """
    if version == "variance" and not (0.0 < r_threshold <= 1.0):
        raise ValueError("r_threshold must be in (0, 1] when version='variance'.")

    mono = deepcopy(finetuned)
    high = deepcopy(finetuned)
    low = deepcopy(finetuned)

    pretrained_params = {
        name: param.detach().clone()
        for name, param in pretrained.named_parameters()
    }
    mono_params = dict(mono.named_parameters())
    high_params = dict(high.named_parameters())
    low_params = dict(low.named_parameters())

    processed = 0
    skipped_not_found = 0
    skipped_shape_mismatch = 0
    skipped_low_update = 0

    with torch.no_grad():
        for name, p_ft in finetuned.named_parameters():
            if name not in pretrained_params:
                skipped_not_found += 1
                continue

            p_pre = pretrained_params[name]
            if p_pre.shape != p_ft.shape:
                skipped_shape_mismatch += 1
                continue

            if not should_process_param(
                weight_pretrained=p_pre,
                weight_finetuned=p_ft,
                min_rel_update=min_rel_update,
            ):
                skipped_low_update += 1
                continue

            w_mono, w_high, w_low = _monosoup_components_for_layer(
                layer_name=name,
                weight_pretrained=p_pre,
                weight_finetuned=p_ft,
                r_threshold=r_threshold,
                version=version,
                verbose=verbose,
            )
            mono_params[name].copy_(w_mono)
            high_params[name].copy_(w_high)
            low_params[name].copy_(w_low)
            processed += 1

    LOGGER.info(
        "MonoSoup ablation summary | processed=%d | skipped_not_found=%d | skipped_shape_mismatch=%d | skipped_low_update=%d",
        processed,
        skipped_not_found,
        skipped_shape_mismatch,
        skipped_low_update,
    )
    return mono, high, low


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
                LOGGER.info("[DIFF] %s: ||dW||_F = %.3e", name, diff_norm)

    LOGGER.info(
        "[DEBUG] Number of params with ||dW||_F > %.1e: %d",
        threshold,
        diff_count,
    )


class LinearCKAAccumulator:
    """
    Streaming linear CKA between feature matrices X and Y with matching shape.
    """

    def __init__(self, feature_dim: int, device: str = "cpu"):
        self.feature_dim = feature_dim
        self.device = device
        self.n = 0
        self.sum_x = torch.zeros(feature_dim, device=device, dtype=torch.float64)
        self.sum_y = torch.zeros(feature_dim, device=device, dtype=torch.float64)
        self.gram_xx = torch.zeros(feature_dim, feature_dim, device=device, dtype=torch.float64)
        self.gram_yy = torch.zeros(feature_dim, feature_dim, device=device, dtype=torch.float64)
        self.gram_xy = torch.zeros(feature_dim, feature_dim, device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.to(self.device, dtype=torch.float64)
        y = y.to(self.device, dtype=torch.float64)
        if x.shape != y.shape:
            raise ValueError(f"CKA update requires identical shapes. Got {x.shape} vs {y.shape}.")
        if x.ndim != 2:
            raise ValueError(f"Expected 2D features [batch, dim], got shape {x.shape}.")

        self.n += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.gram_xx += x.t() @ x
        self.gram_yy += y.t() @ y
        self.gram_xy += x.t() @ y

    @torch.no_grad()
    def compute(self) -> float:
        if self.n == 0:
            return float("nan")

        n = float(self.n)
        mu_x = self.sum_x / n
        mu_y = self.sum_y / n

        x_centered = self.gram_xx - n * torch.outer(mu_x, mu_x)
        y_centered = self.gram_yy - n * torch.outer(mu_y, mu_y)
        xy_centered = self.gram_xy - n * torch.outer(mu_x, mu_y)

        hsic = (xy_centered.square()).sum()
        norm_x = torch.linalg.norm(x_centered)
        norm_y = torch.linalg.norm(y_centered)
        if norm_x == 0 or norm_y == 0:
            return float("nan")
        return float((hsic / (norm_x * norm_y)).item())


class ConvNeXtBlockFeatureExtractor:
    """
    Register forward hooks on ConvNeXtBlock modules and return BxD block features.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: list[Any] = []
        self.features: list[torch.Tensor | None] = []

        blocks = []
        for _, module in model.named_modules():
            if module.__class__.__name__ == "ConvNeXtBlock":
                blocks.append(module)

        if not blocks:
            raise RuntimeError("No ConvNeXtBlock modules found in model.")

        self.num_blocks = len(blocks)
        self.features = [None for _ in range(self.num_blocks)]

        for index, block in enumerate(blocks):
            handle = block.register_forward_hook(self._make_hook(index))
            self.handles.append(handle)

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            if x.ndim == 4:
                feat = x.mean(dim=(2, 3))
            else:
                feat = x.reshape(x.shape[0], -1)
            self.features[index] = feat.detach().cpu()

        return hook

    @torch.inference_mode()
    def extract_batch(self, images: torch.Tensor) -> list[torch.Tensor]:
        self.features = [None for _ in range(self.num_blocks)]
        _ = self.model(images)
        if any(feat is None for feat in self.features):
            raise RuntimeError("Some ConvNeXt block hooks did not receive outputs.")
        return [feat for feat in self.features if feat is not None]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


@torch.inference_mode()
def compute_cka_for_dataset(
    dataset_cls: type,
    dataset_name: str,
    preprocess,
    models: Mapping[str, nn.Module],
    data_location: str,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> Dict[str, list[float]]:
    """
    Compute layer-wise CKA(pretrained, model) for all models in MODEL_ORDER.
    """
    dataset = dataset_cls(preprocess, data_location, batch_size)
    loader = dataset.test_loader

    for model in models.values():
        model.eval()

    extractors = {
        model_name: ConvNeXtBlockFeatureExtractor(model)
        for model_name, model in models.items()
    }
    num_layers = extractors["pretrained"].num_blocks

    cka_acc: Dict[str, list[LinearCKAAccumulator | None]] = {
        model_name: [None for _ in range(num_layers)]
        for model_name in MODEL_ORDER
    }

    LOGGER.info(
        "[CKA] Dataset %s: collecting features for up to %d batches.",
        dataset_name,
        max_batches,
    )

    try:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            batch = maybe_dictionarize_batch(batch)
            images = batch["images"].to(device, non_blocking=True)

            feats_batch: Dict[str, list[torch.Tensor]] = {}
            for model_name, extractor in extractors.items():
                feats_batch[model_name] = extractor.extract_batch(images)

            for layer_idx in range(num_layers):
                x_pre = feats_batch["pretrained"][layer_idx]
                for model_name in MODEL_ORDER:
                    y_model = feats_batch[model_name][layer_idx]
                    if cka_acc[model_name][layer_idx] is None:
                        dim = x_pre.shape[1]
                        cka_acc[model_name][layer_idx] = LinearCKAAccumulator(
                            feature_dim=dim,
                            device="cpu",
                        )
                    cka_acc[model_name][layer_idx].update(x_pre, y_model)

            if batch_idx % 10 == 0:
                LOGGER.info(
                    "[CKA] %s batch %d/%d",
                    dataset_name,
                    batch_idx,
                    max_batches,
                )

    finally:
        for extractor in extractors.values():
            extractor.close()

    cka_scores: Dict[str, list[float]] = {}
    for model_name in MODEL_ORDER:
        scores = []
        for layer_idx in range(num_layers):
            acc = cka_acc[model_name][layer_idx]
            scores.append(float("nan") if acc is None else acc.compute())
        cka_scores[model_name] = scores

    LOGGER.info("[CKA] Finished dataset %s.", dataset_name)
    return cka_scores


def plot_cka_heatmap(
    cka_scores: Mapping[str, Sequence[float]],
    dataset_name: str,
    save_dir: Path,
) -> Path:
    """
    Plot CKA(pretrained, model) per block (columns) and model (rows).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    num_layers = len(cka_scores["pretrained"])
    row_labels = [MODEL_LABELS[name] for name in MODEL_ORDER]
    data = np.stack([np.asarray(cka_scores[name], dtype=np.float64) for name in MODEL_ORDER], axis=0)
    annot_data = np.round(data, 2)

    plt.figure(figsize=(1.8 + num_layers * 0.45, 4.0))
    axis = sns.heatmap(
        data,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        xticklabels=[f"block_{i:02d}" for i in range(num_layers)],
        yticklabels=row_labels,
        annot=annot_data,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "CKA w.r.t. pretrained"},
    )
    axis.set_title(f"CKA(pretrained, ·) on {dataset_name}")
    axis.set_xlabel("ConvNeXt block")
    axis.set_ylabel("Model")
    plt.tight_layout()

    out_path = save_dir / f"cka_{dataset_name}.pdf"
    plt.savefig(out_path, dpi=300)
    plt.close()
    LOGGER.info("[CKA] Saved heatmap for %s to %s", dataset_name, out_path)
    return out_path


def serialize_config(config: RunConfig) -> Dict[str, Any]:
    config_dict = asdict(config)
    for key in ("model_soups_root", "plot_dir", "cka_json_path", "output_json"):
        if config_dict[key] is not None:
            config_dict[key] = str(config_dict[key])
    for key in ("eval_models", "eval_datasets", "cka_datasets"):
        config_dict[key] = list(config_dict[key])
    return config_dict


def default_cka_json_path(config: RunConfig) -> Path:
    DEFAULT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    version = normalize_version(config.version)
    return DEFAULT_ARTIFACT_DIR / f"cka_scores_convnext_{version}_R{config.r_threshold}.json"


def save_cka_results(
    path: Path,
    config: RunConfig,
    cka_results: Mapping[str, Mapping[str, Sequence[float]]],
    eval_results: Mapping[str, Mapping[str, float]],
    plot_paths: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": serialize_config(config),
        "cka_results": cka_results,
        "eval_results": eval_results,
        "plots": dict(plot_paths),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    LOGGER.info("Saved CKA results JSON to: %s", path)


def save_run_summary(
    path: Path,
    config: RunConfig,
    cka_results: Mapping[str, Mapping[str, Sequence[float]]],
    eval_results: Mapping[str, Mapping[str, float]],
    cka_json_path: Path,
    plot_paths: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    eval_metrics: Dict[str, Dict[str, float | None]] = {}
    for model_name, results in eval_results.items():
        imagenet_acc = results.get("ImageNet")
        ood_scores = [v for k, v in results.items() if k != "ImageNet"]
        eval_metrics[model_name] = {
            "imagenet": float(imagenet_acc) if imagenet_acc is not None else None,
            "avg_ood": float(np.mean(ood_scores)) if ood_scores else None,
        }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": serialize_config(config),
        "num_cka_datasets": len(cka_results),
        "eval_metrics": eval_metrics,
        "artifacts": {
            "cka_json": str(cka_json_path),
            "plots": dict(plot_paths),
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    LOGGER.info("Saved run summary to: %s", path)


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Publication-ready ConvNeXt MonoSoup + CKA runner.",
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
        default=0.8,
        help="Variance threshold used only when --version variance.",
    )
    parser.add_argument("--min-rel-update", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-differences", action="store_true")
    parser.add_argument("--verbose-layers", action="store_true")
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run accuracy evaluation for selected models/datasets.",
    )
    parser.add_argument(
        "--eval-models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
        help="Models to evaluate when --run-eval is set.",
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        choices=DATASET_CLASS_NAMES,
        default=list(DATASET_CLASS_NAMES),
        help="Datasets to evaluate when --run-eval is set.",
    )
    parser.add_argument(
        "--cka-datasets",
        nargs="+",
        choices=DATASET_CLASS_NAMES,
        default=["ImageNet", "ImageNetA"],
        help="Datasets used for CKA computation.",
    )
    parser.add_argument(
        "--max-cka-batches",
        type=int,
        default=25,
        help="Maximum number of batches per dataset for CKA.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable CKA heatmap plotting.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "plots",
        help="Directory for CKA heatmaps.",
    )
    parser.add_argument(
        "--cka-json-path",
        type=Path,
        default=None,
        help="Optional output path for full CKA results JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for compact run summary JSON.",
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
        run_eval=args.run_eval,
        eval_models=tuple(args.eval_models),
        eval_datasets=tuple(args.eval_datasets),
        cka_datasets=tuple(args.cka_datasets),
        max_cka_batches=args.max_cka_batches,
        no_plots=args.no_plots,
        plot_dir=args.plot_dir,
        cka_json_path=args.cka_json_path,
        output_json=args.output_json,
        in12k_model_name=args.in12k_model_name,
        in1k_model_name=args.in1k_model_name,
    )


def run(config: RunConfig) -> Dict[str, Dict[str, list[float]]]:
    set_reproducibility(config.seed)
    device = resolve_device(config.device)
    version = normalize_version(config.version)

    LOGGER.info("Resolved device: %s", device)
    LOGGER.info("Run config: %s", json.dumps(serialize_config(config), indent=2, sort_keys=True))

    dataset_classes = import_model_soups_datasets(config.model_soups_root)
    pretrained_model, finetuned_model, preprocess = load_pretrained_and_finetuned_convnext(
        device=device,
        in12k_model_name=config.in12k_model_name,
        in1k_model_name=config.in1k_model_name,
    )

    if config.debug_differences:
        debug_param_differences(
            pretrained_model=pretrained_model,
            finetuned_model=finetuned_model,
            threshold=1e-6,
            max_print=60,
        )

    monosoup_model, high_model, low_model = apply_monosoup_with_ablation(
        pretrained=pretrained_model,
        finetuned=finetuned_model,
        r_threshold=config.r_threshold,
        min_rel_update=config.min_rel_update,
        version=version,
        verbose=config.verbose_layers,
    )

    models = {
        "pretrained": pretrained_model,
        "finetuned": finetuned_model,
        "monosoup": monosoup_model,
        "highonly": high_model,
        "lowonly": low_model,
    }

    eval_results: Dict[str, Dict[str, float]] = {}
    if config.run_eval:
        for model_name in config.eval_models:
            eval_results[model_name] = eval_model(
                model=models[model_name],
                preprocess=preprocess,
                dataset_names=config.eval_datasets,
                dataset_classes=dataset_classes,
                data_location=config.data_location,
                batch_size=config.batch_size,
                device=device,
                model_name=f"ConvNeXt {MODEL_LABELS[model_name]}",
            )
    else:
        LOGGER.info("Skipping evaluation (--run-eval not set).")

    cka_results: Dict[str, Dict[str, list[float]]] = {}
    plot_paths: Dict[str, str] = {}

    for dataset_name in config.cka_datasets:
        cka_scores = compute_cka_for_dataset(
            dataset_cls=dataset_classes[dataset_name],
            dataset_name=dataset_name,
            preprocess=preprocess,
            models=models,
            data_location=config.data_location,
            batch_size=config.batch_size,
            device=device,
            max_batches=config.max_cka_batches,
        )
        cka_results[dataset_name] = cka_scores

        if not config.no_plots:
            plot_path = plot_cka_heatmap(
                cka_scores=cka_scores,
                dataset_name=dataset_name,
                save_dir=config.plot_dir,
            )
            plot_paths[dataset_name] = str(plot_path)

    cka_json_path = config.cka_json_path or default_cka_json_path(config)
    save_cka_results(
        path=cka_json_path,
        config=config,
        cka_results=cka_results,
        eval_results=eval_results,
        plot_paths=plot_paths,
    )

    if config.output_json is not None:
        save_run_summary(
            path=config.output_json,
            config=config,
            cka_results=cka_results,
            eval_results=eval_results,
            cka_json_path=cka_json_path,
            plot_paths=plot_paths,
        )

    return cka_results


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    config = parse_args(argv)
    cka_results = run(config)
    LOGGER.info("Completed CKA for datasets: %s", ", ".join(cka_results.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
