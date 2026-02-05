# CLI and Reproducibility

Date: 2026-02-05

This note explains how `MonoSoup.py` and `CKA.py` are structured for reproducible runs and portable experiment artifacts.

## 1) Structured Run Configuration

Both scripts define a frozen `RunConfig` dataclass and parse CLI arguments into it.

- `MonoSoup.py`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L61>
- `CKA.py`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L93>

This keeps every experimental setting explicit and serializable.

## 2) Deterministic Runtime Controls

Both scripts call `set_reproducibility(seed)` before model construction and evaluation:

- Python RNG seed,
- NumPy seed,
- Torch CPU and CUDA seeds,
- deterministic CuDNN setup,
- deterministic algorithms mode where supported.

Code links:

- `MonoSoup.py`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L124>
- `CKA.py`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L132>

## 3) Device Resolution

`--device auto` selects CUDA if available, else CPU.

Code links:

- `MonoSoup.py`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L142>
- `CKA.py`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L147>

## 4) Dataset Interface Contract

Both scripts import dataset wrappers from a local `model-soups` repository and expect classes such as:

- `ImageNet`,
- `ImageNetV2`,
- `ImageNetSketch`,
- `ImageNetR`,
- `ObjectNet`,
- `ImageNetA`.

Code links:

- `MonoSoup.py`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L152>
- `CKA.py`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L155>

Batch normalization helper for dataloader outputs:

- `maybe_dictionarize_batch` in `MonoSoup.py`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L178>
- `maybe_dictionarize_batch` in `CKA.py`: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L179>

## 5) Artifact Export

### MonoSoup

`--output-json` stores:

- timestamp,
- run config,
- per-dataset accuracy,
- compact metrics (`imagenet`, `avg_ood`).

Code: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/MonoSoup.py#L637>

### CKA

`CKA.py` writes:

- full CKA results JSON,
- optional heatmaps,
- optional compact run summary JSON.

Code links:

- full results: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L938>
- summary: <https://github.com/alirezaabdollahpour/MonoSoup/blob/main/CKA.py#L958>

## 6) Suggested Experiment Logging Discipline

For stable comparisons across runs:

1. Fix `--seed`.
2. Keep `--max-cka-batches` constant.
3. Log `--version`, `--R`, and `--min-rel-update`.
4. Preserve JSON artifacts in versioned result folders.
5. Pair CKA similarity with classification accuracy in analysis.

## 7) Example Commands

MonoSoup:

```bash
python MonoSoup.py \
  --pretrained-checkpoint /path/to/model_0.pt \
  --finetuned-checkpoint /path/to/model_31.pt \
  --data-location /path/to/data_root \
  --model-type 32 \
  --version freevariance \
  --seed 0 \
  --output-json results/monosoup_clip.json
```

CKA:

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
  --seed 0 \
  --output-json results/cka_clip_summary.json
```
