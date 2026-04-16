#!/usr/bin/env python3
"""
17_biomedclip_linear_probe.py

Linear-probe classification for BOTH OpenAI CLIP and BiomedCLIP on the four
neuroimaging datasets.  Extends 16_multimodal_clip_classification.py by running
linear probing for every loaded model (not only OpenAI-CLIP), enabling a fair
head-to-head comparison of frozen visual features across architectures.

Usage:
    python 17_biomedclip_linear_probe.py                        # all datasets
    python 17_biomedclip_linear_probe.py --datasets MRI_tumor_binary_norm
    python 17_biomedclip_linear_probe.py --models OpenAI-CLIP BiomedCLIP

    NOTE: OpenAI-CLIP linear probe results are already filled in (from 16_multimodal_clip_classification.py). Only the BiomedCLIP linear probe rows (marked †)
    are missing. To fill those in without re-running CLIP, use:

        python 17_biomedclip_linear_probe.py --models BiomedCLIP

Results are written to:
    results/multimodal/linear_probe_both_models_results.csv
    results/multimodal/linear_probe_both_models_report.json

Dependencies:
    pip install open_clip_torch ftfy regex tqdm pillow transformers
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
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
    RESULTS_DIR    = os.path.join(BASE_DIR, "results", "multimodal")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "multimodal")

    BATCH_SIZE             = 32
    NUM_EPOCHS_LINEAR      = 20
    LEARNING_RATE_LINEAR   = 1e-3
    WEIGHT_DECAY           = 1e-5
    PATIENCE               = 5
    MIN_DELTA              = 1e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASETS = [
        "MRI_tumor_binary_norm",
        "MRI_tumor_multiclass_norm",
        "MRI_ms_norm",
        "CT_stroke_binary_norm",
    ]

    # Class prompts (same as script 16 for consistency)
    CLASS_PROMPTS = {
        "MRI_tumor_binary_norm": {
            "normal": "A normal brain MRI scan without any tumor or abnormality",
            "tumor":  "A brain MRI scan showing a brain tumor or mass lesion",
        },
        "MRI_tumor_multiclass_norm": {
            "Carcinoma":      "An MRI scan showing carcinoma metastatic to the brain",
            "Germinoma":      "An MRI scan showing a germinoma brain tumor",
            "Glioma":         "An MRI scan showing a glioma brain tumor",
            "Granuloma":      "An MRI scan showing a granuloma in the brain",
            "Meduloblastoma": "An MRI scan showing a medulloblastoma tumor",
            "Meningioma":     "An MRI scan showing a meningioma brain tumor",
            "Neurocitoma":    "An MRI scan showing a neurocytoma tumor",
            "Normal":         "A normal brain MRI scan without tumor",
            "Other":          "An MRI scan showing other brain abnormality",
            "Papiloma":       "An MRI scan showing a papilloma in the brain",
            "Schwannoma":     "An MRI scan showing a schwannoma brain tumor",
            "Ttuberculoma":   "An MRI scan showing a tuberculoma in the brain",
        },
        "MRI_ms_norm": {
            "Control": "A normal brain MRI FLAIR sequence without multiple sclerosis lesions",
            "MS":      "A brain MRI FLAIR sequence showing multiple sclerosis lesions and plaques",
        },
        "CT_stroke_binary_norm": {
            "normal": "A normal brain CT scan without stroke or ischemia",
            "stroke": "A brain CT scan showing acute ischemic stroke or hemorrhage",
        },
    }

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
        self.dataset_name = dataset_name
        self.split        = split
        self.transform    = transform
        self.split_dir    = os.path.join(config.SPLIT_DIR, dataset_name, split)

        self.data         = []
        self.class_names  = []
        self.class_to_idx = {}
        self._load_data()

        print(f"[Dataset] {dataset_name}/{split}: "
              f"{len(self.data)} images, classes={self.class_names}")

    def _load_data(self):
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Split directory not found: {self.split_dir}")

        class_dirs = sorted(
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        )
        for idx, class_name in enumerate(class_dirs):
            self.class_names.append(class_name)
            self.class_to_idx[class_name] = idx
            class_path = os.path.join(self.split_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.data.append({
                        "path":      os.path.join(class_path, img_name),
                        "class":     class_name,
                        "class_idx": idx,
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        image = Image.open(item["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "class": item["class"],
                "class_idx": item["class_idx"], "path": item["path"]}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_models(model_names: list[str]) -> dict:
    """Load requested vision-language models."""
    models = {}

    if "OpenAI-CLIP" in model_names:
        print("Loading OpenAI CLIP (ViT-B/32)...")
        clip_model, _, clip_pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        clip_model = clip_model.to(config.DEVICE)
        clip_tok   = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(config.DEVICE)
            clip_dim = clip_model.encode_image(dummy).shape[1]
        models["OpenAI-CLIP"] = {
            "model": clip_model, "tokenizer": clip_tok, "preprocess": clip_pre,
            "embed_dim": clip_dim,
        }
        print(f"  OpenAI CLIP loaded  (embed_dim={clip_dim})")

    if "BiomedCLIP" in model_names:
        print("Loading BiomedCLIP...")
        try:
            bm_id  = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            bm_m, _, bm_pre = open_clip.create_model_and_transforms(bm_id)
            bm_m   = bm_m.to(config.DEVICE)
            bm_tok = open_clip.get_tokenizer(bm_id)
            with torch.no_grad():
                dummy  = torch.randn(1, 3, 224, 224).to(config.DEVICE)
                bm_dim = bm_m.encode_image(dummy).shape[1]
            models["BiomedCLIP"] = {
                "model": bm_m, "tokenizer": bm_tok, "preprocess": bm_pre,
                "embed_dim": bm_dim,
            }
            print(f"  BiomedCLIP loaded  (embed_dim={bm_dim})")
        except Exception as e:
            print(f"  BiomedCLIP not available: {e}")

    return models

# ─────────────────────────────────────────────────────────────────────────────
# LINEAR PROBE
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(model, loader) -> tuple:
    """One-shot feature extraction from a frozen backbone.

    Runs the vision encoder exactly once over the split and caches the
    results as CPU tensors, eliminating redundant forward passes during
    the linear-probe training loop.
    """
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Extracting features"):
            f = F.normalize(model.encode_image(batch["image"].to(config.DEVICE)), dim=-1)
            feats.append(f.cpu())
            labs.append(batch["class_idx"])
    return torch.cat(feats), torch.cat(labs)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def linear_probe_train(model_info: dict, train_loader, val_loader,
                       num_classes: int, model_name: str,
                       dataset_name: str) -> tuple:
    print(f"\n{'='*60}")
    print(f"Linear Probe: {model_name} on {dataset_name}")
    print(f"{'='*60}")

    model = model_info["model"]
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Pre-extract features once — backbone is frozen so features never change
    print("  Pre-extracting train features...")
    train_feats, train_labels = _extract_features(model, train_loader)
    print("  Pre-extracting val features...")
    val_feats, val_labels     = _extract_features(model, val_loader)

    from torch.utils.data import TensorDataset
    train_dl = DataLoader(TensorDataset(train_feats, train_labels),
                          batch_size=config.BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(TensorDataset(val_feats,   val_labels),
                          batch_size=config.BATCH_SIZE, shuffle=False)

    classifier = LinearClassifier(model_info["embed_dim"], num_classes).to(config.DEVICE)
    optimizer  = optim.AdamW(classifier.parameters(),
                             lr=config.LEARNING_RATE_LINEAR,
                             weight_decay=config.WEIGHT_DECAY)
    criterion  = nn.CrossEntropyLoss()
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc     = 0.0
    patience_counter = 0
    best_state       = None

    for epoch in range(config.NUM_EPOCHS_LINEAR):
        # Train
        classifier.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for features, labels in tqdm(train_dl,
                                     desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS_LINEAR} [Train]"):
            features = features.to(config.DEVICE)
            labels   = labels.to(config.DEVICE)
            outputs  = classifier(features)
            loss     = criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            t_correct += outputs.argmax(1).eq(labels).sum().item()
            t_total   += labels.size(0)

        t_loss /= len(train_dl)
        t_acc   = t_correct / t_total

        # Val
        classifier.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in tqdm(val_dl,
                                         desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS_LINEAR} [Val]"):
                features = features.to(config.DEVICE)
                labels   = labels.to(config.DEVICE)
                outputs  = classifier(features)
                v_loss  += criterion(outputs, labels).item()
                v_correct += outputs.argmax(1).eq(labels).sum().item()
                v_total   += labels.size(0)

        v_loss /= len(val_dl)
        v_acc   = v_correct / v_total

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        scheduler.step(v_acc)

        print(f"Epoch {epoch+1}: train_loss={t_loss:.4f} train_acc={t_acc:.4f} "
              f"val_loss={v_loss:.4f} val_acc={v_acc:.4f}")

        if v_acc > best_val_acc + config.MIN_DELTA:
            best_val_acc     = v_acc
            patience_counter = 0
            best_state       = classifier.state_dict()
            # Save checkpoint
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"linear_probe_{model_name}_{dataset_name}_best.pt")
            torch.save(best_state, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        classifier.load_state_dict(best_state)
    print(f"Best val acc: {best_val_acc:.4f}")
    return classifier, history


def linear_probe_eval(model_info: dict, classifier, test_loader,
                      model_name: str, dataset_name: str) -> dict:
    model = model_info["model"]
    model.eval()
    classifier.eval()

    print("  Pre-extracting test features...")
    test_feats, test_labels = _extract_features(model, test_loader)

    from torch.utils.data import TensorDataset
    test_dl = DataLoader(TensorDataset(test_feats, test_labels),
                         batch_size=config.BATCH_SIZE, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for features, labels in tqdm(test_dl, desc="Linear probe test"):
            features = features.to(config.DEVICE)
            outputs  = classifier(features)
            probs    = F.softmax(outputs, dim=1)
            preds    = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return _compute_metrics(all_preds, all_labels, all_probs,
                            model_name, dataset_name, "linear-probe")

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(preds, labels, probs, model_name, dataset_name,
                     method) -> dict:
    preds  = np.array(preds)
    labels = np.array(labels)
    probs  = np.array(probs)
    n      = probs.shape[1]

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds,
                  average="binary" if n == 2 else "weighted",
                  zero_division=0)
    try:
        auc = (roc_auc_score(labels, probs[:, 1])
               if n == 2
               else roc_auc_score(labels, probs,
                                  multi_class="ovr", average="weighted"))
    except Exception as e:
        print(f"  AUC not computed: {e}")
        auc = None

    if auc is not None:
        print(f"  {model_name} | {dataset_name} | {method}: "
              f"Acc={accuracy:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    else:
        print(f"  {model_name} | {dataset_name} | {method}: "
              f"Acc={accuracy:.4f}  F1={f1:.4f}  AUC=N/A")

    return {
        "model":       model_name,
        "dataset":     dataset_name,
        "method":      method,
        "accuracy":    float(accuracy),
        "f1":          float(f1),
        "auc":         float(auc) if auc is not None else None,
        "num_classes": n,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print("LINEAR PROBE COMPARISON — SUMMARY")
    print(f"{'='*70}")
    print(df.to_string(index=False))

    avg = df.groupby("model")[["accuracy", "f1", "auc"]].mean()
    print("\nAverage by model:")
    print(avg.to_string())

    best = df.loc[df.groupby("dataset")["accuracy"].idxmax()]
    print("\nBest model per dataset:")
    print(best[["dataset", "model", "accuracy", "f1"]].to_string(index=False))


def save_results(df: pd.DataFrame):
    csv_path = os.path.join(config.RESULTS_DIR,
                            "linear_probe_both_models_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    report = {
        "experiment": "Linear Probe — CLIP vs BiomedCLIP",
        "date":       pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models":     df["model"].unique().tolist(),
        "datasets":   df["dataset"].unique().tolist(),
        "results":    df.to_dict("records"),
    }
    json_path = os.path.join(config.RESULTS_DIR,
                             "linear_probe_both_models_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {json_path}")

    # Bar-chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ["accuracy", "f1", "auc"]):
        try:
            pivot = df.pivot(index="dataset", columns="model", values=metric)
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{metric.upper()} by Dataset", fontsize=12, fontweight="bold")
            ax.set_ylabel(metric.capitalize())
            ax.set_xlabel("")
            ax.legend(title="Model")
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
        except Exception as e:
            ax.set_title(f"{metric} — {e}")
    plt.suptitle("Linear Probe: OpenAI CLIP vs BiomedCLIP", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(config.RESULTS_DIR,
                            "linear_probe_both_models_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(datasets: list[str], model_names: list[str]):
    all_models = load_models(model_names)
    if not all_models:
        print("No models loaded. Exiting.")
        return

    lp_results = []

    for dataset in datasets:
        print(f"\n\n{'#'*70}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*70}")

        num_classes = len(config.CLASS_PROMPTS[dataset])

        # Run linear probing for every loaded model
        for model_name, model_info in all_models.items():
            pre = model_info["preprocess"]

            # These loaders are used only for one-shot feature extraction;
            # shuffling is unnecessary here (handled inside TensorDataset loaders).
            train_dl = DataLoader(
                MedicalImageDataset(dataset, "train", pre),
                batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            val_dl   = DataLoader(
                MedicalImageDataset(dataset, "val",   pre),
                batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            test_dl  = DataLoader(
                MedicalImageDataset(dataset, "test",  pre),
                batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

            classifier, _ = linear_probe_train(
                model_info, train_dl, val_dl, num_classes, model_name, dataset)
            result = linear_probe_eval(
                model_info, classifier, test_dl, model_name, dataset)
            lp_results.append(result)

    df = pd.DataFrame(lp_results)
    if not df.empty:
        print_summary(df)
        save_results(df)
    else:
        print("No results collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear probing for CLIP and BiomedCLIP on neuroimaging datasets")
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS,
                        help="Datasets to evaluate")
    parser.add_argument("--models",   nargs="+",
                        default=["OpenAI-CLIP", "BiomedCLIP"],
                        choices=["OpenAI-CLIP", "BiomedCLIP"],
                        help="Models to probe (default: both)")
    args = parser.parse_args()
    main(datasets=args.datasets, model_names=args.models)
