# MonoSoup

Publication-ready implementations of MonoSoup experiments for CLIP and ConvNeXt, including reproducible evaluation and ConvNeXt CKA analysis.

## What This Repository Contains

This repository currently focuses on three production-ready scripts:

1. `MonoSoup.py`
2. `Convnext_freevariance_vetterli.py`
3. `CKA_ConvNext.py`

All three scripts use explicit CLI arguments, deterministic runtime setup, structured logging, and JSON artifact export for reproducibility.

## Method Summary

MonoSoup edits weights layer-wise between a pretrained model `W0` and a fine-tuned model `W1`:

1. Compute `delta = W1 - W0`.
2. Apply SVD to `delta` per layer.
3. Split into `delta_high` and `delta_low`.
4. Mix both components using MonoSoup coefficients.
5. Construct edited weights.

Supported selection modes:

1. `variance`: fixed spectral energy threshold `R`.
2. `vetterli` / `freevariance`: per-layer effective-rank rule (Roy-Vetterli style).

## Environment

Recommended Python version:

1. Python 3.10+

Main dependencies:

1. `torch`
2. `torchvision`
3. `timm`
4. `open-clip-torch`
5. `numpy`
6. `matplotlib`
7. `seaborn`

Install example:

```bash
pip install torch torchvision timm open-clip-torch numpy matplotlib seaborn
```

## Required External Dataset Wrapper

The scripts depend on dataset classes from the `model-soups` repository and expect a local path to it.

Default expected location:

1. `./model-soups`

You can override this with `--model-soups-root`.

## Data Location Convention

Pass `--data-location` as the root used by `model-soups` dataset classes.

Example directory layout:

```text
/data_root/
  imagenet/
  imagenetv2-matched-frequency-format-val/
  imagenet-a/
  imagenet-r/
  objectnet-1.0/
  sketch/
```

## Script Guide

### `MonoSoup.py`

Purpose:

1. Applies MonoSoup between two CLIP checkpoints.
2. Evaluates on ImageNet and OOD datasets from `model-soups`.

Input requirements:

1. `--pretrained-checkpoint`: checkpoint for `W0`.
2. `--finetuned-checkpoint`: checkpoint for `W1`.
3. Checkpoints must include `classification_head.weight` and `classification_head.bias`.

Typical command:

```bash
python MonoSoup.py \
  --pretrained-checkpoint /path/to/model_0.pt \
  --finetuned-checkpoint /path/to/model_31.pt \
  --data-location /path/to/data_root \
  --model-type 32 \
  --version freevariance \
  --device auto \
  --output-json results/monosoup_clip.json
```

Key outputs:

1. Console/log metrics per dataset.
2. Optional JSON report via `--output-json`.

### `Convnext_freevariance_vetterli.py`

Purpose:

1. Builds a paired ConvNeXt setup automatically using timm pretrained weights.
2. Applies MonoSoup on ConvNeXt.
3. Optionally evaluates on ID/OOD datasets.
4. Exports per-layer lambda statistics.

Typical command:

```bash
python Convnext_freevariance_vetterli.py \
  --data-location /path/to/data_root \
  --version vetterli \
  --R 0.9 \
  --device auto \
  --output-json results/monosoup_convnext.json
```

Key outputs:

1. Layer statistics JSON (default under `ConvNext/`).
2. Optional run summary JSON via `--output-json`.

### `CKA_ConvNext.py`

Purpose:

1. Builds ConvNeXt model variants: `pretrained`, `finetuned`, `monosoup`, `highonly`, `lowonly`.
2. Computes block-wise linear CKA with respect to the pretrained model.
3. Optionally evaluates classification accuracy.
4. Saves CKA JSON artifacts and optional heatmaps.

Typical command:

```bash
python CKA_ConvNext.py \
  --data-location /path/to/data_root \
  --version vetterli \
  --R 0.8 \
  --cka-datasets ImageNet ImageNetA \
  --max-cka-batches 25 \
  --device auto \
  --run-eval \
  --output-json results/cka_convnext_summary.json
```

Key outputs:

1. Full CKA JSON (default under `ConvNext/cka/`).
2. Optional PDF heatmaps (unless `--no-plots`).
3. Optional compact run summary JSON.

## Reproducibility Notes

All scripts:

1. Set seeds for Python, NumPy, and PyTorch.
2. Enable deterministic CuDNN behavior.
3. Record run configuration in JSON outputs when enabled.

## Quick Start Workflow

1. Clone this repository.
2. Install dependencies.
3. Clone `model-soups` locally and pass `--model-soups-root` if needed.
4. Run one script with explicit `--data-location`.
5. Save artifacts with `--output-json` (and script-specific artifact flags).

## Citation

If this code is used in your research, please cite the corresponding MonoSoup paper and this repository.
