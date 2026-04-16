#!/usr/bin/env python3
"""
18_layer_fusion_benchmark.py

Layer-wise feature extraction and multi-layer fusion benchmark for BiomedCLIP
(ViT-B/16) on four neuroimaging datasets.

Replaces 18_Multimodal_benchmark_layer_fusion.ipynb with an optimised script:
  - BiomedCLIP loaded ONCE
  - Features at layers 2, 6, 11 captured in a SINGLE backbone pass per split,
    then cached as CPU tensors — no repeated ViT forward passes during training
  - All five classifier configs trained on the cached tensors

Five configs:
  CLIP-Layer2          shallow  (≈25 % depth)
  CLIP-Layer6          middle   (≈50 % depth)
  CLIP-Layer11         deep     (pre-final LN, ≈100 % depth)
  CLIP-Fusion-Concat   concat of all three layers → MLP
  CLIP-Fusion-Weighted learned weighted sum → MLP

Fixes vs notebook:
  - Layer indices corrected for ViT-B/16 (12 blocks, 0–11);
    notebook used 3/18/23 designed for ViT-L/14 (24 blocks)
  - Single BiomedCLIP load shared across all configs
  - Feature caching eliminates O(configs × epochs) backbone calls
  - KeyError 'model' in summary fixed

Usage:
    python 18_layer_fusion_benchmark.py                      # all datasets
    python 18_layer_fusion_benchmark.py --datasets MRI_ms_norm

Results:
    results/layer_fusion_benchmark/layer_fusion_results.csv
    results/layer_fusion_benchmark/layer_fusion_report.json
    results/layer_fusion_benchmark/layer_fusion_comparison.png

Dependencies:
    pip install open_clip_torch ftfy regex tqdm pillow scikit-learn
"""

import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device:     {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    BASE_DIR       = os.environ.get("THESIS_DIR",
                     os.path.expanduser("~/Documents/MSc_Thesis_Neuroimaging"))
    SPLIT_DIR      = os.path.join(BASE_DIR, "data", "split")
    RESULTS_DIR    = os.path.join(BASE_DIR, "results", "layer_fusion_benchmark")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "layer_fusion")

    BATCH_SIZE   = 32
    NUM_EPOCHS   = 20
    LR           = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE     = 5
    MIN_DELTA    = 1e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BiomedCLIP is ViT-B/16: 12 transformer blocks (indices 0–11), embed 768
    EMBED_DIM     = 768
    SHALLOW_LAYER = 2    # ≈ 25 % depth
    MIDDLE_LAYER  = 6    # ≈ 50 % depth
    DEEP_LAYER    = 11   # ≈100 % depth (before final LayerNorm)
    LAYER_INDICES = [2, 6, 11]

    DATASETS = [
        "MRI_tumor_binary_norm",
        "MRI_tumor_multiclass_norm",
        "MRI_ms_norm",
        "CT_stroke_binary_norm",
    ]

    def __init__(self):
        os.makedirs(self.RESULTS_DIR,    exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)


config = Config()
print(f"Results dir: {config.RESULTS_DIR}")
print(f"Device:      {config.DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MedicalImageDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, transform=None):
        self.transform = transform
        self.split_dir = os.path.join(config.SPLIT_DIR, dataset_name, split)

        self.data        = []
        self.class_names = []
        self._load_data()
        print(f"  [{dataset_name}/{split}] {len(self.data)} images, "
              f"classes={self.class_names}")

    def _load_data(self):
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Split directory not found: {self.split_dir}")
        class_dirs = sorted(
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        )
        for idx, cls in enumerate(class_dirs):
            self.class_names.append(cls)
            cls_path = os.path.join(self.split_dir, cls)
            for img in os.listdir(cls_path):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.data.append({"path": os.path.join(cls_path, img),
                                      "class_idx": idx})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        image = Image.open(item["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "class_idx": item["class_idx"]}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_biomedclip():
    print("Loading BiomedCLIP (ViT-B/16)...")
    model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, _, preprocess = open_clip.create_model_and_transforms(model_id)
    model = model.to(config.DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    num_blocks = len(model.visual.transformer.resblocks)
    print(f"  Blocks: {num_blocks}  embed_dim: {config.EMBED_DIM}")
    assert max(config.LAYER_INDICES) < num_blocks, (
        f"Layer index {max(config.LAYER_INDICES)} out of range "
        f"for {num_blocks}-block ViT")
    return model, preprocess

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (single pass, all layers)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_layer_features(model, loader,
                                layer_indices: list) -> tuple[dict, torch.Tensor]:
    """
    One forward pass through the frozen ViT, capturing the CLS token at each
    requested layer index simultaneously.

    Returns
    -------
    layer_feats : dict {layer_idx: Tensor(N, 768)}  L2-normalised, CPU
    labels      : Tensor(N,)                        CPU
    """
    layer_set   = set(layer_indices)
    accumulated = {i: [] for i in layer_indices}
    all_labels  = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Extracting layers {layer_indices}"):
            images = batch["image"].to(config.DEVICE)
            all_labels.append(batch["class_idx"])

            # Manual ViT-B/16 forward (mirrors open_clip VisionTransformer)
            x = model.visual.conv1(images)            # (B, C, H, W)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, C, N_patches)
            x = x.permute(0, 2, 1)                   # (B, N_patches, C)

            # Prepend CLS token and add positional embedding
            cls = (model.visual.class_embedding.to(x.dtype)
                   + torch.zeros(x.shape[0], 1, x.shape[-1],
                                 dtype=x.dtype, device=x.device))
            x = torch.cat([cls, x], dim=1)            # (B, N_patches+1, C)
            x = x + model.visual.positional_embedding.to(x.dtype)
            x = model.visual.ln_pre(x)

            # Walk transformer blocks, snapshotting requested layers
            for i, block in enumerate(model.visual.transformer.resblocks):
                x = block(x)
                if i in layer_set:
                    # x[:, 0, :] = CLS token in NLD (batch-first) format
                    accumulated[i].append(
                        F.normalize(x[:, 0, :].float(), dim=-1).cpu())

    layer_feats = {i: torch.cat(accumulated[i]) for i in layer_indices}
    labels      = torch.cat(all_labels)
    return layer_feats, labels

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────

def _mlp_head(input_dim: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


class SingleLayerClassifier(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = config.EMBED_DIM):
        super().__init__()
        self.head = _mlp_head(embed_dim, num_classes)

    def forward(self, feats: list):
        return self.head(feats[0])


class FusionConcatClassifier(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = config.EMBED_DIM,
                 num_layers: int = 3):
        super().__init__()
        self.head = _mlp_head(embed_dim * num_layers, num_classes)

    def forward(self, feats: list):
        return self.head(torch.cat(feats, dim=-1))


class FusionWeightedClassifier(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = config.EMBED_DIM,
                 num_layers: int = 3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.head    = _mlp_head(embed_dim, num_classes)

    def forward(self, feats: list):
        w     = F.softmax(self.weights, dim=0)
        fused = sum(w[i] * feats[i] for i in range(len(feats)))
        return self.head(fused)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _build_loader(tensors: tuple, shuffle: bool) -> DataLoader:
    """Wrap pre-cached feature tensors in a DataLoader."""
    return DataLoader(TensorDataset(*tensors),
                      batch_size=config.BATCH_SIZE, shuffle=shuffle)


def train_classifier(classifier: nn.Module,
                     train_tensors: tuple, val_tensors: tuple,
                     cfg_name: str, dataset_name: str) -> nn.Module:
    """
    Train classifier on pre-cached feature tensors.
    The first N-1 elements of each tuple are feature tensors; the last is labels.
    """
    train_dl  = _build_loader(train_tensors, shuffle=True)
    val_dl    = _build_loader(val_tensors,   shuffle=False)
    optimizer = optim.AdamW(classifier.parameters(),
                            lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc     = 0.0
    patience_counter = 0
    best_state       = None

    for epoch in range(config.NUM_EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        classifier.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for *feat_tensors, labels in train_dl:
            feats  = [f.to(config.DEVICE) for f in feat_tensors]
            labels = labels.to(config.DEVICE)
            out    = classifier(feats)
            loss   = criterion(out, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            t_correct += out.argmax(1).eq(labels).sum().item()
            t_total   += labels.size(0)
        t_loss /= len(train_dl)
        t_acc   = t_correct / t_total

        # ── Val ────────────────────────────────────────────────────────────
        classifier.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for *feat_tensors, labels in val_dl:
                feats  = [f.to(config.DEVICE) for f in feat_tensors]
                labels = labels.to(config.DEVICE)
                out    = classifier(feats)
                v_loss    += criterion(out, labels).item()
                v_correct += out.argmax(1).eq(labels).sum().item()
                v_total   += labels.size(0)
        v_loss /= len(val_dl)
        v_acc   = v_correct / v_total

        scheduler.step(v_acc)
        print(f"    Epoch {epoch+1:2d}: "
              f"train_acc={t_acc:.4f}  val_acc={v_acc:.4f}  val_loss={v_loss:.4f}")

        if v_acc > best_val_acc + config.MIN_DELTA:
            best_val_acc     = v_acc
            patience_counter = 0
            best_state       = {k: v.clone()
                                for k, v in classifier.state_dict().items()}
            ckpt = os.path.join(config.CHECKPOINT_DIR,
                                f"lfb_{cfg_name}_{dataset_name}_best.pt")
            torch.save(best_state, ckpt)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        classifier.load_state_dict(best_state)
    print(f"  Best val acc: {best_val_acc:.4f}")
    return classifier


def eval_classifier(classifier: nn.Module, test_tensors: tuple,
                    cfg_name: str, dataset_name: str) -> dict:
    test_dl = _build_loader(test_tensors, shuffle=False)
    classifier.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for *feat_tensors, labels in test_dl:
            feats = [f.to(config.DEVICE) for f in feat_tensors]
            out   = classifier(feats)
            probs = F.softmax(out, dim=1)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    n      = probs.shape[1]
    avg    = "binary" if n == 2 else "weighted"

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average=avg, zero_division=0)
    rec  = recall_score(labels, preds,    average=avg, zero_division=0)
    f1   = f1_score(labels, preds,        average=avg, zero_division=0)
    try:
        auc = (roc_auc_score(labels, probs[:, 1]) if n == 2
               else roc_auc_score(labels, probs,
                                  multi_class="ovr", average="weighted"))
    except Exception as e:
        print(f"    AUC skipped: {e}")
        auc = None

    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    print(f"  {cfg_name} | {dataset_name}: "
          f"Acc={acc:.4f}  F1={f1:.4f}  AUC={auc_str}")

    return {
        "model":       cfg_name,
        "dataset":     dataset_name,
        "accuracy":    float(acc),
        "precision":   float(prec),
        "recall":      float(rec),
        "f1":          float(f1),
        "auc":         float(auc) if auc is not None else None,
        "num_classes": n,
    }

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("LAYER FUSION BENCHMARK — SUMMARY")
    print(f"{'='*70}")
    print(df[["model", "dataset", "accuracy", "f1", "auc"]].to_string(index=False))

    print("\nAverage by model:")
    print(df.groupby("model")[["accuracy", "f1", "auc"]].mean().to_string())

    print("\nBest config per dataset:")
    best = df.loc[df.groupby("dataset")["accuracy"].idxmax()]
    print(best[["dataset", "model", "accuracy", "f1"]].to_string(index=False))


def save_results(df: pd.DataFrame):
    csv_path = os.path.join(config.RESULTS_DIR, "layer_fusion_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    report = {
        "experiment":    "BiomedCLIP Layer-wise & Fusion Benchmark",
        "date":          pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "layer_indices": config.LAYER_INDICES,
        "models":        df["model"].unique().tolist(),
        "datasets":      df["dataset"].unique().tolist(),
        "results":       df.to_dict("records"),
    }
    json_path = os.path.join(config.RESULTS_DIR, "layer_fusion_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {json_path}")

    # Bar-chart comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, metric in zip(axes.flat, ["accuracy", "f1", "precision", "recall"]):
        try:
            pivot = df.pivot(index="dataset", columns="model", values=metric)
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(metric.capitalize(), fontsize=12, fontweight="bold")
            ax.set_ylabel(metric.capitalize())
            ax.set_xlabel("")
            ax.legend(title="Config", fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
        except Exception as e:
            ax.set_title(f"{metric} — {e}")

    plt.suptitle("BiomedCLIP Layer-wise vs Fusion (ViT-B/16)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(config.RESULTS_DIR, "layer_fusion_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# (cfg_name, layer_key, is_fusion, fusion_type)
# layer_key: int for single-layer configs (index into layer_feats dict)
#            None for fusion configs (all layers used)
MODEL_CONFIGS = [
    (f"CLIP-Layer{config.SHALLOW_LAYER}", config.SHALLOW_LAYER, False, None),
    (f"CLIP-Layer{config.MIDDLE_LAYER}",  config.MIDDLE_LAYER,  False, None),
    (f"CLIP-Layer{config.DEEP_LAYER}",    config.DEEP_LAYER,    False, None),
    ("CLIP-Fusion-Concat",                None,                  True,  "concat"),
    ("CLIP-Fusion-Weighted",              None,                  True,  "weighted"),
]


def _build_tensors(layer_feats: dict, labels: torch.Tensor,
                   layer_key, is_fusion: bool) -> tuple:
    """Pack pre-extracted feature tensors into the format expected by DataLoader."""
    if is_fusion:
        return tuple(layer_feats[i] for i in config.LAYER_INDICES) + (labels,)
    return (layer_feats[layer_key], labels)


def main(datasets: list):
    model, preprocess = load_biomedclip()
    all_results = []

    for dataset in datasets:
        print(f"\n\n{'#'*70}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*70}")

        train_dir   = os.path.join(config.SPLIT_DIR, dataset, "train")
        num_classes = len([d for d in os.listdir(train_dir)
                           if os.path.isdir(os.path.join(train_dir, d))])
        print(f"  Classes: {num_classes}")

        # ── Step 1: extract features for all splits in one pass each ────────
        print("\n[1/3] Extracting features...")
        split_feats  = {}
        split_labels = {}
        for split in ("train", "val", "test"):
            loader = DataLoader(
                MedicalImageDataset(dataset, split, preprocess),
                batch_size=config.BATCH_SIZE, shuffle=False,
                num_workers=4, pin_memory=True)
            split_feats[split], split_labels[split] = extract_all_layer_features(
                model, loader, config.LAYER_INDICES)

        # ── Step 2: train & eval each classifier config ──────────────────────
        print("\n[2/3] Training classifiers...")
        for cfg_name, layer_key, is_fusion, fusion_type in MODEL_CONFIGS:
            print(f"\n  {'='*50}")
            print(f"  Config: {cfg_name}")
            print(f"  {'='*50}")

            train_t = _build_tensors(split_feats["train"], split_labels["train"],
                                     layer_key, is_fusion)
            val_t   = _build_tensors(split_feats["val"],   split_labels["val"],
                                     layer_key, is_fusion)
            test_t  = _build_tensors(split_feats["test"],  split_labels["test"],
                                     layer_key, is_fusion)

            if not is_fusion:
                clf = SingleLayerClassifier(num_classes).to(config.DEVICE)
            elif fusion_type == "concat":
                clf = FusionConcatClassifier(num_classes).to(config.DEVICE)
            else:
                clf = FusionWeightedClassifier(num_classes).to(config.DEVICE)

            clf    = train_classifier(clf, train_t, val_t, cfg_name, dataset)
            result = eval_classifier(clf, test_t, cfg_name, dataset)
            all_results.append(result)

    # ── Step 3: save ─────────────────────────────────────────────────────────
    print("\n[3/3] Saving results...")
    df = pd.DataFrame(all_results)
    if not df.empty:
        print_summary(df)
        save_results(df)
    else:
        print("No results collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BiomedCLIP layer-wise & fusion benchmark on neuroimaging datasets")
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS,
                        help="Datasets to benchmark (default: all four)")
    args = parser.parse_args()
    main(datasets=args.datasets)
