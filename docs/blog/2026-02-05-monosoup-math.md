# MonoSoup Mathematics

Date: 2026-02-05

This article explains how `MonoSoup.py` constructs an edited model between a pretrained checkpoint and a fine-tuned checkpoint.

Key implementation entry points:

- `apply_monosoup`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L507>
- `_monosoup_update_for_layer`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L424>
- `_choose_k_and_pk`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L379>

## 1) Problem Setup

For each trainable layer, define:

$$
\Delta W = W_1 - W_0,
$$

where:

- $W_0$: pretrained weights,
- $W_1$: fine-tuned weights.

`MonoSoup.py` first checks whether a parameter should be processed (`should_process_param`) using a relative update test:

$$
\frac{\|\Delta W\|_F}{\|W_0\|_F + \epsilon} \ge \tau.
$$

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L350>

## 2) Matrix View for SVD

SVD is applied to a 2D view of each tensor:

- linear: shape `[out, in]`,
- convolution: reshape to `[out, in * k_h * k_w]`.

Code path:

- reshape in: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L331>
- reshape out: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L346>

## 3) Spectral Decomposition

`_monosoup_update_for_layer` computes:

$$
\Delta W = U \Sigma V^\top.
$$

Given $k$, the update is split into:

$$
\Delta W_{\text{high}} = U_{1:k}\Sigma_{1:k}V_{1:k}^\top,
\qquad
\Delta W_{\text{low}} = \Delta W - \Delta W_{\text{high}}.
$$

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L470>

## 4) Choosing k: Two Modes

### Variance Mode

Use the smallest $k$ such that cumulative squared singular-value energy reaches threshold $R$:

$$
\frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} \ge R.
$$

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L413>

### Freevariance Mode (Roy-Vetterli style effective rank)

Build normalized singular-value magnitudes:

$$
p_i = \frac{|\sigma_i|}{\sum_j |\sigma_j|}.
$$

Compute entropy and effective rank:

$$
H = -\sum_i p_i \log(p_i + \epsilon), \qquad
r_{\text{eff}} = e^H.
$$

Set:

$$
k = \lceil r_{\text{eff}} \rceil.
$$

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L397>

## 5) MonoSoup Mixing Coefficients

After selecting $k$, define:

$$
P_k = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2},
\qquad
\rho = \left(\frac{\sigma_{k+1}}{\sigma_1 + \epsilon}\right)^2.
$$

Then:

$$
\cos(\alpha) = \sqrt{1 - P_k},
$$

$$
\lambda_{\text{low}} = \rho + (1-\rho)\cos(\alpha),
\qquad
\lambda_{\text{high}} = 1 - \lambda_{\text{low}}.
$$

The edited update:

$$
\Delta W_{\text{mono}}
= \lambda_{\text{high}}\Delta W_{\text{high}}
+ \lambda_{\text{low}}\Delta W_{\text{low}}.
$$

Final layer:

$$
W_{\text{mono}} = W_0 + \Delta W_{\text{mono}}.
$$

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L476>

## 6) Full-Model Pass

`apply_monosoup` iterates over matched parameters and applies the layer update where valid:

- skips not-found keys,
- skips shape mismatch,
- skips very small updates.

A summary of processed and skipped layers is logged.

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L562>

## 7) Practical Complexity

The dominant cost per processed layer is compact SVD of the flattened update matrix. For large layers, this can be expensive in both memory and time. In practice:

- `min_rel_update` reduces unnecessary decomposition work,
- `verbose_layers` helps inspect spectral behavior,
- float16/bfloat16 tensors are cast to float32 before SVD for stability.

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L449>

## 8) Minimal Reproducible Command

```bash
python MonoSoup.py \
  --pretrained-checkpoint /path/to/model_0.pt \
  --finetuned-checkpoint /path/to/model_31.pt \
  --data-location /path/to/data_root \
  --model-type 32 \
  --version freevariance \
  --R 0.8 \
  --output-json results/monosoup_clip.json
```
