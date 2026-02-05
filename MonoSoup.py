"""
MonoSoup for CLIP checkpoints with reproducible evaluation.

This script:
1. Loads a pretrained checkpoint (W0) and a fine-tuned checkpoint (W1).
2. Applies layer-wise MonoSoup editing to build an edited model.
3. Evaluates the edited model on ImageNet + OOD datasets from model-soups.

Usage example:
python MonoSoup_freevariance_vetterli.py \
  --pretrained-checkpoint /path/to/model_0.pt \
  --finetuned-checkpoint /path/to/model_31.pt \
  --data-location /path/to/imagenet \
  --version freevariance \
  --model-type 32 \
  --device auto \
  --output-json results/monosoup_freevariance.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import open_clip
import torch
import torch.nn as nn


LOGGER = logging.getLogger("monosoup")

SUPPORTED_OPEN_CLIP_MODELS = {
    "32": "ViT-B-32",
    "16": "ViT-B-16",
    "14": "ViT-L-14",
}

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
    pretrained_checkpoint: Path
    finetuned_checkpoint: Path
    data_location: str
    model_soups_root: Path = Path("../model-soups")
    model_type: str = "32"
    device: str = "auto"
    version: str = "freevariance"
    r_threshold: float = 0.8
    min_rel_update: float = 1e-6
    batch_size: int = 256
    seed: int = 0
    verbose_layers: bool = False
    debug_differences: bool = False
    output_json: Path | None = None


class ModelWrapper(nn.Module):
    """Wrap open_clip visual encoder with a linear classification head."""

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        num_classes: int,
        normalize: bool = False,
        initial_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalize = normalize
        self.classification_head = nn.Linear(feature_dim, num_classes)

        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))

        self.classification_head.weight = nn.Parameter(initial_weights.clone())
        self.classification_head.bias = nn.Parameter(
            torch.zeros_like(self.classification_head.bias)
        )

        if hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images: torch.Tensor, return_features: bool = False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Keep training/eval behavior deterministic where kernels support it.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Backward compatibility with older torch versions.
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

    Returns:
        list[type]: [ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]
    """
    model_soups_root = model_soups_root.resolve()
    if not model_soups_root.exists():
        raise FileNotFoundError(f"model-soups root not found: {model_soups_root}")

    root_str = str(model_soups_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    datasets_module = importlib.import_module("datasets")
    missing = [name for name in DATASET_CLASS_NAMES if not hasattr(datasets_module, name)]
    if missing:
        raise ImportError(
            "Could not import required dataset classes from model-soups/datasets.py. "
            f"Missing: {missing}. Imported module: {datasets_module.__file__}"
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


def get_model_from_sd(
    state_dict: Mapping[str, torch.Tensor],
    base_model: nn.Module,
    device: torch.device,
) -> nn.Module:
    if "classification_head.weight" not in state_dict:
        raise KeyError(
            "Checkpoint is missing 'classification_head.weight'. "
            "Expected a model-soups linear-probe checkpoint."
        )
    if "classification_head.bias" not in state_dict:
        raise KeyError(
            "Checkpoint is missing 'classification_head.bias'. "
            "Expected a model-soups linear-probe checkpoint."
        )

    feature_dim = state_dict["classification_head.weight"].shape[1]
    num_classes = state_dict["classification_head.weight"].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    for param in model.parameters():
        param.data = param.data.float()
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def load_open_clip_model_and_preprocess(
    model_type: str,
    device: torch.device,
) -> Tuple[nn.Module, Any]:
    """
    Load an open_clip model and validation transform.

    model_type:
      "32" -> ViT-B-32
      "16" -> ViT-B-16
      "14" -> ViT-L-14
    """
    if model_type not in SUPPORTED_OPEN_CLIP_MODELS:
        supported = ", ".join(sorted(SUPPORTED_OPEN_CLIP_MODELS))
        raise ValueError(f"Unknown model_type '{model_type}'. Supported values: {supported}")

    model_name = SUPPORTED_OPEN_CLIP_MODELS[model_type]
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained="openai",
        device=device.type,
    )
    return model, preprocess_val


@torch.inference_mode()
def test_model_on_dataset(model: nn.Module, dataset, device: torch.device) -> float:
    """
    Evaluate model on a single model-soups dataset.
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
        targets = labels

        logits = model(inputs)

        projection_fn = getattr(dataset, "project_logits", None)
        if projection_fn is not None:
            logits = projection_fn(logits, device.type)

        if hasattr(dataset, "project_labels"):
            targets = dataset.project_labels(targets, device.type)

        if isinstance(logits, list):
            logits = logits[0]

        pred = logits.argmax(dim=1, keepdim=True).to(device)
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
    model_name: str = "MonoSoup",
) -> Dict[str, float]:
    """
    Evaluate model on ImageNet and OOD datasets.
    """
    results: Dict[str, float] = {}
    for dataset_cls in dataset_classes:
        LOGGER.info("[%s] Evaluating on %s", model_name, dataset_cls.__name__)
        dataset = dataset_cls(preprocess, data_location, batch_size)
        accuracy = test_model_on_dataset(model, dataset, device=device)
        LOGGER.info("[%s] %s accuracy: %.4f", model_name, dataset_cls.__name__, accuracy)
        results[dataset_cls.__name__] = accuracy

    id_acc = results["ImageNet"]
    ood_scores = [value for key, value in results.items() if key != "ImageNet"]
    mean_ood = float(np.mean(ood_scores)) if ood_scores else float("nan")
    LOGGER.info("[%s] In-distribution (ImageNet): %.4f", model_name, id_acc)
    LOGGER.info("[%s] Avg OOD: %.4f", model_name, mean_ood)
    return results


def _reshape_to_matrix(weight: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Flatten a tensor to a 2D matrix for SVD:
      - Linear weights already have shape [out, in].
      - Conv weights become [out, in * k_h * k_w].
    """
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
    """
    Decide whether to apply MonoSoup to this parameter.

    Rules:
      - Parameter must be at least 2D.
      - Relative update norm must be larger than min_rel_update.
    """
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
    """
    Select how many singular values to keep for one layer.

    Returns:
      num_keep: number of singular directions to keep (>=1)
      k_index : 0-based index of the last kept singular value
      p_k     : captured energy fraction
    """
    rank = singular_values.shape[0]

    if version == "freevariance":
        spectrum_l1 = singular_values.abs()
        total_mag = spectrum_l1.sum()

        if total_mag < eps:
            num_keep = 1
        else:
            probs = spectrum_l1 / (total_mag + eps)
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

    raise ValueError(f"Unknown version '{version}'. Use 'variance' or 'freevariance'.")


def _monosoup_update_for_layer(
    layer_name: str,
    weight_pretrained: torch.Tensor,
    weight_finetuned: torch.Tensor,
    r_threshold: float,
    version: str,
    eps: float = 1e-12,
    verbose: bool = False,
) -> torch.Tensor:
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

    # SVD is more stable in float32 if checkpoints are half precision.
    svd_dtype = torch.float32 if delta.dtype in (torch.float16, torch.bfloat16) else delta.dtype
    delta_svd = delta.to(svd_dtype)
    u, singular_values, v_h = torch.linalg.svd(delta_svd, full_matrices=False)

    singular_values_sq = singular_values.square()
    total_energy = singular_values_sq.sum()
    if total_energy < eps:
        if verbose:
            LOGGER.info("[MonoSoup] Layer '%s': spectral energy ~ 0, copied from W1.", layer_name)
        return weight_finetuned.clone()

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
    edited_matrix = w0_matrix.to(svd_dtype) + delta_mono
    edited_weight = _reshape_from_matrix(edited_matrix, original_shape)
    return edited_weight.to(device=device, dtype=output_dtype)


def apply_monosoup(
    pretrained: nn.Module,
    finetuned: nn.Module,
    r_threshold: float = 0.8,
    min_rel_update: float = 1e-6,
    version: str = "variance",
    verbose: bool = False,
) -> nn.Module:
    """
    Apply layer-wise MonoSoup to all matching 2D+ parameters.
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

            new_weight = _monosoup_update_for_layer(
                layer_name=name,
                weight_pretrained=p_pre,
                weight_finetuned=p_edit,
                r_threshold=r_threshold,
                version=version,
                verbose=verbose,
            )
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
    """
    Log which parameters differ between pretrained and fine-tuned models.
    """
    LOGGER.info("[DEBUG] Checking parameter differences between checkpoints.")
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


def load_state_dict(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    loaded = torch.load(path, map_location=device)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Expected checkpoint mapping at {path}, got {type(loaded)}")

    # Accept checkpoints stored as {"state_dict": ...}.
    state_dict = loaded["state_dict"] if "state_dict" in loaded and isinstance(loaded["state_dict"], Mapping) else loaded

    # Accept DataParallel checkpoints with "module." prefixes.
    if "classification_head.weight" not in state_dict and any(
        key.startswith("module.") for key in state_dict.keys()
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    if not isinstance(state_dict, dict):
        state_dict = dict(state_dict)
    return state_dict


def serialize_config(config: RunConfig) -> Dict[str, Any]:
    config_dict = asdict(config)
    for key in ("pretrained_checkpoint", "finetuned_checkpoint", "model_soups_root", "output_json"):
        if config_dict[key] is not None:
            config_dict[key] = str(config_dict[key])
    return config_dict


def save_results(output_path: Path, config: RunConfig, results: Mapping[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": serialize_config(config),
        "results": dict(results),
        "metrics": {
            "imagenet": results.get("ImageNet"),
            "avg_ood": float(
                np.mean([v for k, v in results.items() if k != "ImageNet"])
            ) if any(k != "ImageNet" for k in results) else None,
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    LOGGER.info("Saved results to %s", output_path)


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Apply MonoSoup between CLIP checkpoints and evaluate on model-soups datasets.",
    )
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-location",
        type=str,
        required=True,
        help="Root path used by model-soups dataset classes.",
    )
    parser.add_argument(
        "--model-soups-root",
        type=Path,
        default=Path("../model-soups"),
        help="Path to the local model-soups repository.",
    )
    parser.add_argument(
        "--model-type",
        choices=sorted(SUPPORTED_OPEN_CLIP_MODELS),
        default="32",
        help="open_clip architecture alias: 32=ViT-B-32, 16=ViT-B-16, 14=ViT-L-14.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="'auto' picks CUDA when available.",
    )
    parser.add_argument(
        "--version",
        choices=("variance", "freevariance"),
        default="freevariance",
        help="MonoSoup variant.",
    )
    parser.add_argument(
        "--R",
        type=float,
        default=0.8,
        help="Variance threshold for version='variance'. Ignored for freevariance.",
    )
    parser.add_argument("--min-rel-update", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-differences", action="store_true")
    parser.add_argument(
        "--verbose-layers",
        action="store_true",
        help="Log MonoSoup statistics for each processed layer.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for serialized config + results JSON.",
    )

    args = parser.parse_args(argv)
    return RunConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        finetuned_checkpoint=args.finetuned_checkpoint,
        data_location=args.data_location,
        model_soups_root=args.model_soups_root,
        model_type=args.model_type,
        device=args.device,
        version=args.version,
        r_threshold=args.R,
        min_rel_update=args.min_rel_update,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose_layers=args.verbose_layers,
        debug_differences=args.debug_differences,
        output_json=args.output_json,
    )


def run(config: RunConfig) -> Dict[str, float]:
    set_reproducibility(config.seed)
    device = resolve_device(config.device)
    LOGGER.info("Resolved device: %s", device)
    LOGGER.info("Run config: %s", json.dumps(serialize_config(config), indent=2, sort_keys=True))

    dataset_classes = import_model_soups_datasets(config.model_soups_root)
    base_model, preprocess = load_open_clip_model_and_preprocess(
        model_type=config.model_type,
        device=device,
    )

    pre_sd = load_state_dict(config.pretrained_checkpoint, device=device)
    ft_sd = load_state_dict(config.finetuned_checkpoint, device=device)

    pretrained_model = get_model_from_sd(pre_sd, deepcopy(base_model), device=device)
    finetuned_model = get_model_from_sd(ft_sd, deepcopy(base_model), device=device)

    if config.debug_differences:
        debug_param_differences(
            pretrained_model=pretrained_model,
            finetuned_model=finetuned_model,
            threshold=1e-6,
            max_print=80,
        )

    edited_model = apply_monosoup(
        pretrained=pretrained_model,
        finetuned=finetuned_model,
        r_threshold=config.r_threshold,
        min_rel_update=config.min_rel_update,
        version=config.version,
        verbose=config.verbose_layers,
    )

    results = eval_model(
        edited_model,
        preprocess=preprocess,
        dataset_classes=dataset_classes,
        data_location=config.data_location,
        batch_size=config.batch_size,
        device=device,
        model_name=f"MonoSoup({config.version})",
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    config = parse_args(argv)
    results = run(config)

    if config.output_json is not None:
        save_results(config.output_json, config, results)

    LOGGER.info("Final results: %s", json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
