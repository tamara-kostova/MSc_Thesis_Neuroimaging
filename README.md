# MSc Thesis Neuroimaging

Exploratory notebooks and scripts for an MSc thesis benchmarking image classifiers
on neuroimaging data. The work spans three families of models over the same four
tasks, so that results are directly comparable across them:

1. **Supervised CNNs** — eight ImageNet-pretrained backbones fine-tuned per task
   (ResNet50/101, VGG16, DenseNet121/169, MobileNetV2, EfficientNet-B0/B4).
2. **Vision-language models** — CLIP and BiomedCLIP, evaluated zero-shot, by linear
   probe on frozen features, and with multi-layer feature fusion. MedGemma 1.5 and
   Chameleon are evaluated as generative classifiers.
3. **Segmentation-derived features** — SAM 3, explored for zero-shot segmentation
   quality and as a frozen feature extractor for linear probing.

On top of these sit explainability, calibration, and uncertainty experiments
(Grad-CAM, attention rollout, temperature/Platt/isotonic scaling, conformal
prediction, MC dropout, deep ensembles, TTA).

### The four tasks

| Task | Modality | Classes | Sources |
|---|---|---|---|
| `MRI_tumor_binary` | MRI | tumour / no tumour | Figshare, Br35H |
| `MRI_tumor_multiclass` | MRI | 14 tumour subtypes | Kaggle 17-class + 44-class, MD5-deduplicated |
| `MRI_ms` | MRI | MS / control | Kaggle multiple-sclerosis |
| `CT_stroke_binary` | CT | stroke / no stroke | AISD, Kaggle brain-stroke-CT |

`datasets_split.md` documents each source and the splitting plan. Splits are
leakage-safe: the Figshare tumour set is split by patient ID, the multiclass set is
MD5-deduplicated across its two sources first, and everything else is stratified
70/15/15 (`05_data_splitting.py`).

`MRI_ms` is the hard one — accuracies sit near chance (46.6–59.7% at 20 epochs).
That is a finding, not a bug, and several scripts special-case it accordingly.

### Notebooks and scripts

Numbered roughly in execution order. Notebooks are Colab-oriented; several were
later rewritten as standalone `.py` scripts for long or resumable runs, and where
both exist the script is the one that was actually run.

| File | What it does |
|---|---|
| `01_data_download_and_exploration.ipynb` | Downloads the source datasets and does first-pass inspection. |
| `02_data_preprocessing.ipynb` | Grayscale conversion and intensity standardization. |
| `03_data_loaders_augmentation.ipynb` | `Dataset` class and augmentation policy. |
| `05_data_splitting.py` | Leakage-safe train/val/test splitting for all four tasks. |
| `04_models_training.ipynb` | First CNN benchmark, `EPOCHS = 5`. Superseded. |
| `04_models_training_checkpoint.ipynb` / `.py` | The CNN benchmark of record, `NUM_EPOCHS = 20`, with checkpoint/resume across Colab disconnects. Also runs the McNemar significance tests between model pairs. |
| `05_model_comparison.ipynb` | Cross-model comparison tables. **Reads the 5-epoch JSON** and writes `results/THESIS_TABLE.md` + `results/model_summary.csv`, so its output is stale relative to the 20-epoch benchmark below. |
| `06_sam3_exploration.ipynb`, `07_sam3_adapter_finetune.ipynb` | SAM 3 setup and adapter fine-tuning. |
| `12`–`14_SAM3_*.ipynb` | SAM 3 segmentation evaluated per dataset (MS, tumour incl. BraTS2021, stroke). `_all_slices` variants sweep every slice rather than a selected one. |
| `15_SAM3_linearn_probing.ipynb` | Linear probes on frozen SAM 3 features. |
| `16_multimodal_clip_classification.py` / `.ipynb` | CLIP and BiomedCLIP, zero-shot and linear probe. |
| `17_biomedclip_linear_probe.py` | Extends 16 so every model gets a linear probe, for a fair head-to-head on frozen features. |
| `17_Chameleon_*.ipynb` | Chameleon as a generative classifier. |
| `18_layer_fusion_benchmark.py` / `.ipynb` | BiomedCLIP layer-wise extraction and multi-layer fusion; the script caches one backbone pass instead of re-running the ViT per config. |
| `08`–`11_medgemma*.ipynb` | MedGemma 1.5 4B: loading, metadata generation, and evaluation. |
| `19_Uncertainty_*.ipynb` | Uncertainty and calibration experiments. |
| `cnns_explanability.py` | Explainability/calibration/uncertainty for the eight CNNs, with per-architecture guards (e.g. MC dropout only where dropout layers exist). |
| `multimodal_explanability.py` | The same for CLIP/BiomedCLIP, ViT-aware (attention rollout, head-only MC dropout over the frozen backbone). |
| `uncertainty_confidence.py` | Shared calibration/conformal/attribution implementations. |
| `results/make_benchmark_artifacts.py` | Rebuilds the benchmark table and figures from a results file. |

Dependencies are split by track: `requirements_cnn.txt` for the CNN work,
`requirements_clip.txt` for the vision-language work.

## CNN benchmark artifacts (`results/benchmarks/`)

There are two CNN benchmark runs in this project's history. Only one of them is
authoritative.

| Artifact | Run | Status |
|---|---|---|
| `results/benchmarks/summary_20260208_214002.csv` | 20 epochs (`NUM_EPOCHS = 20`), from `04_models_training_checkpoint.py` | **Canonical.** Single source of truth. These are the numbers reported in the MIPRO paper and the MSc thesis. |
| `benchmark_results_5epoch_20260110_212727.json` | 5 epochs (`EPOCHS = 5`), from `04_models_training.ipynb` | **Superseded.** Kept for reference. |

### Layout

```
results/benchmarks/
  summary_20260208_214002.csv   20-epoch results
  FINAL_TABLE.md                20-epoch table
  20epoch/
    heatmap_20.png              20-epoch figures
    model_ranking_20.png
    param_efficiency_20.png
    task_difficulty_20.png
  5epoch/
    heatmap_5.png               5-epoch figures
    model_ranking_5.png
    param_efficiency_5.png
    task_difficulty_5.png
```

One directory per run. `FINAL_TABLE.md` and the CSV stay at the top level because
there is only one of each.

**The numeric suffix is the epoch count.** The original `*_20.png` family came
from the `NUM_EPOCHS = 20` run in `04_models_training_checkpoint.ipynb`; the
5-epoch notebook never produced figures with that suffix. `make_benchmark_artifacts.py`
derives the suffix from `--epochs`, so a 5-epoch render lands on `_5` and cannot
overwrite the `_20` set. Each figure's title states its epoch count as
well, so the two sets are distinguishable by filename, by directory, and on sight.

`FINAL_TABLE.md` and all eight PNGs are **derived** artifacts. Do not hand-edit
them — regenerate them from the run data:

```bash
# 20-epoch: table at the top level, figures in 20epoch/
python results/make_benchmark_artifacts.py results/benchmarks/summary_20260208_214002.csv \
    --out-dir results/benchmarks --only table --epochs 20
python results/make_benchmark_artifacts.py results/benchmarks/summary_20260208_214002.csv \
    --out-dir results/benchmarks/20epoch --only figures --epochs 20

# 5-epoch figures
python results/make_benchmark_artifacts.py benchmark_results_5epoch_20260110_212727.json \
    --out-dir results/benchmarks/5epoch --only figures --epochs 5
```

The script reads either a summary CSV or a raw `benchmark_results_*.json`.

That script is standalone: it reads only the results file and rewrites the table
and the figures. It does not train anything and does not touch `checkpoints/`. Before
writing, it refuses to proceed if any binary task scores below chance, if any
metric falls outside `[0, 1]`, or if any model/dataset cell is missing.

### Metric definitions

From `04_models_training_checkpoint.py:619-629`:

- **accuracy** — top-1 accuracy.
- **F1**, **precision**, **recall** — `average='weighted'`.
- **AUC** — ROC-AUC. Binary on the two-class tasks (`MRI_tumor_binary`, `MRI_ms`,
  `CT_stroke_binary`); one-vs-rest (`multi_class='ovr'`) on `MRI_tumor_multiclass`.

These are the definitions behind every number in `FINAL_TABLE.md`.
