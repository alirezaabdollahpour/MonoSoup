# CKA Deep Dive

Date: 2026-02-05

This article explains how `CKA.py` computes block-wise representation similarity between:

- `pretrained`,
- `finetuned`,
- `monosoup`,
- `highonly`,
- `lowonly`.

Core implementation links:

- `apply_monosoup_with_ablation`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L559>
- `LinearCKAAccumulator`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L664>
- `BlockFeatureExtractor`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L716>
- `compute_cka_for_dataset`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L792>

## 1) Model Variants

`CKA.py` first builds three edited models from the same layer-wise SVD decomposition:

- MonoSoup: $W_0 + \Delta W_{\text{mono}}$
- High-only: $W_0 + \Delta W_{\text{high}}$
- Low-only: $W_0 + \Delta W_{\text{low}}$

This design isolates which spectral components drive representation shifts.

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L469>

## 2) Feature Extraction at Transformer Blocks

Features are collected by registering forward hooks on visual transformer blocks.

For each mini-batch and each model, the extractor returns a list:

$$
\{F^{(1)}, F^{(2)}, \dots, F^{(L)}\},
\qquad
F^{(\ell)} \in \mathbb{R}^{B \times D_\ell}.
$$

The hook handles both common tensor layouts:

- `(B, T, D)`,
- `(T, B, D)`.

and reduces token dimension via mean pooling for a stable `B x D` feature matrix.

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L747>

## 3) Linear CKA Definition

Given centered feature matrices $X_c, Y_c \in \mathbb{R}^{n \times d}$:

$$
\mathrm{CKA}(X, Y)
= \frac{\|X_c^\top Y_c\|_F^2}
{\|X_c^\top X_c\|_F \cdot \|Y_c^\top Y_c\|_F}.
$$

`LinearCKAAccumulator` computes this in streaming form by storing sufficient statistics:

- $\sum x_i$, $\sum y_i$,
- $X^\top X$, $Y^\top Y$, $X^\top Y$.

Then centering is reconstructed as:

$$
X_c^\top X_c = X^\top X - n\mu_x\mu_x^\top,
\qquad
Y_c^\top Y_c = Y^\top Y - n\mu_y\mu_y^\top,
$$

$$
X_c^\top Y_c = X^\top Y - n\mu_x\mu_y^\top.
$$

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L696>

## 4) Dataset-Level Loop

For each selected dataset:

1. Load dataset wrapper from `model-soups`.
2. For each batch (up to `max_cka_batches`), extract features for all models.
3. Update one CKA accumulator per layer and model, always against pretrained features.
4. Compute final per-layer scores.

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L828>

## 5) Visualization and Artifacts

`plot_cka_heatmap` renders a row-per-model, column-per-block heatmap with values in `[0, 1]`.

Saved outputs include:

- full CKA JSON (`cka_json_path` or default path),
- optional PDF heatmaps (`plot_dir`),
- optional run summary JSON (`output_json`).

Code links:

- plotting: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L876>
- JSON save: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L938>

## 6) Reading the Curves Scientifically

Typical interpretation pattern:

- High CKA close to 1.0 means layer representations remain close to pretrained geometry.
- Lower CKA indicates stronger representational drift induced by fine-tuning or editing.
- Comparing `monosoup`, `highonly`, and `lowonly` across depth reveals where spectral components dominate adaptation.

This should be interpreted jointly with accuracy (`--run-eval`) to avoid over-reading similarity alone.

## 7) Minimal Reproducible Command

```bash
python CKA.py \
  --pretrained-checkpoint /path/to/model_0.pt \
  --finetuned-checkpoint /path/to/model_31.pt \
  --data-location /path/to/data_root \
  --model-type 32 \
  --version freevariance \
  --cka-datasets ImageNet ImageNetA \
  --max-cka-batches 25 \
  --run-eval \
  --output-json results/cka_clip_summary.json
```
