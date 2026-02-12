"""
Architecture-Aware Explainability, Confidence & Uncertainty for CNN Models
on Medical Imaging (Neuroimaging) Datasets.

Methods implemented (with architectural guards):
  - Calibration: Temperature Scaling, Platt Scaling, Isotonic Regression  (all models)
  - Conformal Prediction (APS)                                           (all models)
  - MC Dropout   -- ONLY for architectures with dropout layers           (VGG16, MobileNet, EfficientNet)
  - Deep Ensembles  -- uses all 8 trained CNN checkpoints                (all models)
  - Test-Time Augmentation (TTA)                                         (all models)
  - Grad-CAM     -- per-architecture target layer mapping                (all models)

Models: resnet50, resnet101, vgg16, densenet121, densenet169,
        mobilenet_v2, efficientnet_b0, efficientnet_b4

Datasets: MRI_tumor_binary_norm, MRI_tumor_multiclass_norm,
          MRI_ms_norm, CT_stroke_binary_norm
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from PIL import Image
import cv2
import shutil
import warnings

warnings.filterwarnings("ignore")

# Reuse calibration / uncertainty primitives from the shared toolkit
from uncertainty_confidence import (
    ConformalPrediction,
    TemperatureScaling,
    PlattScaling,
    IsotonicRegressionCalibration,
    CalibrationMetrics,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = "/mnt/gdrive/MSc_Thesis_Neuroimaging"
    SPLIT_DIR = os.path.join(BASE_DIR, "data/split")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    RESULTS_DIR = os.path.join(BASE_DIR, "results/cnn_explainability")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    DATASETS = {
        "MRI_tumor_binary_norm": {
            "path": os.path.join(SPLIT_DIR, "MRI_tumor_binary_norm"),
        },
        "MRI_tumor_multiclass_norm": {
            "path": os.path.join(SPLIT_DIR, "MRI_tumor_multiclass_norm"),
        },
        "MRI_ms_norm": {
            "path": os.path.join(SPLIT_DIR, "MRI_ms_norm"),
        },
        "CT_stroke_binary_norm": {
            "path": os.path.join(SPLIT_DIR, "CT_stroke_binary_norm"),
        },
    }

    MODELS = [
        "resnet50",
        "resnet101",
        "vgg16",
        "densenet121",
        "densenet169",
        "mobilenet_v2",
        "efficientnet_b0",
        "efficientnet_b4",
    ]


# ============================================================================
# ARCHITECTURE REGISTRIES  (the core of "methods must match architecture")
# ============================================================================

# --- MC Dropout eligibility ---
# Only models whose *standard PyTorch implementation* contains Dropout layers
# that can be meaningfully activated at test time.
# ResNet and DenseNet have NO dropout → MC Dropout is inappropriate.
MC_DROPOUT_ELIGIBLE = {
    "resnet50": False,      # no dropout layers in torchvision ResNet
    "resnet101": False,
    "vgg16": True,          # Dropout(0.5) at classifier[2] and classifier[5]
    "densenet121": False,   # no dropout layers in torchvision DenseNet
    "densenet169": False,
    "mobilenet_v2": True,   # Dropout(0.2) in classifier[0]
    "efficientnet_b0": True,  # Dropout in classifier[0]
    "efficientnet_b4": True,
}

# --- Grad-CAM target layer per architecture ---
# Each lambda receives the model and returns the nn.Module whose output
# activations / gradients are used to build the class activation map.
GRADCAM_TARGET_LAYERS = {
    "resnet50":       lambda m: m.layer4[-1],
    "resnet101":      lambda m: m.layer4[-1],
    "vgg16":          lambda m: m.features[-1],        # last ReLU in conv stack
    "densenet121":    lambda m: m.features.denseblock4, # last dense block
    "densenet169":    lambda m: m.features.denseblock4,
    "mobilenet_v2":   lambda m: m.features[-1],        # last InvertedResidual
    "efficientnet_b0": lambda m: m.features[-1],       # last MBConv block
    "efficientnet_b4": lambda m: m.features[-1],
}


# ============================================================================
# MODEL CREATION & LOADING  (mirrors notebook 04)
# ============================================================================

def create_model(model_name: str, num_classes: int, pretrained: bool = False):
    """Instantiate a torchvision model and swap its classification head."""

    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "densenet169":
        model = models.densenet169(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def load_trained_model(model_name: str, dataset_name: str,
                       num_classes: int, device: torch.device) -> nn.Module:
    """Load a trained CNN checkpoint from disk."""
    ckpt_path = os.path.join(
        Config.CHECKPOINT_DIR, f"{model_name}_{dataset_name}_final.pt"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = create_model(model_name, num_classes, pretrained=False)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# DATASET  (mirrors notebook 04)
# ============================================================================

class MedicalImageDataset(Dataset):
    """PyTorch Dataset for medical images with stratified splits."""

    def __init__(self, split_dir, split_type="train", transform=None):
        self.split_dir = split_dir
        self.split_type = split_type
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self._build_samples()

    def _build_samples(self):
        split_path = os.path.join(self.split_dir, self.split_type)
        idx = 0
        for class_name in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = idx
                idx += 1
            label = self.class_to_idx[class_name]
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(class_path, img_name), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        image_rgb = Image.new("RGB", image.size)
        image_rgb.paste(image)
        if self.transform:
            image_rgb = self.transform(image_rgb)
        return image_rgb, label


def get_data_loaders(split_dir: str, batch_size: int = 32, num_workers: int = 2):
    """Create train / val / test DataLoaders."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    loaders = {}
    class_to_idx = None
    for split in ("train", "val", "test"):
        ds = MedicalImageDataset(split_dir, split, test_transform)
        if class_to_idx is None:
            class_to_idx = ds.class_to_idx
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    return loaders, class_to_idx


# ============================================================================
# LOGIT COLLECTION
# ============================================================================

def collect_logits(model: nn.Module, loader: DataLoader,
                   device: torch.device):
    """Forward-pass a trained CNN and return raw logits + labels."""
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting logits", leave=False):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


# ============================================================================
# EXPERIMENT 1: CALIBRATION  (all models)
# ============================================================================

def run_calibration(model_name, dataset_name, val_logits, val_labels,
                    test_logits, test_labels, output_dir):
    """Temperature / Platt / Isotonic calibration + reliability diagrams."""
    print(f"\n  [Calibration] {model_name}")
    cal_dir = output_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- Baseline (uncalibrated) ---
    baseline_probs = F.softmax(test_logits, dim=-1)
    results["baseline"] = {
        "ece": CalibrationMetrics.expected_calibration_error(baseline_probs, test_labels),
        "mce": CalibrationMetrics.maximum_calibration_error(baseline_probs, test_labels),
        "brier": CalibrationMetrics.brier_score(baseline_probs, test_labels),
    }

    # --- Temperature Scaling ---
    ts = TemperatureScaling()
    temp = ts.calibrate(val_logits, val_labels)
    ts_probs = ts.apply(test_logits)
    results["temperature_scaling"] = {
        "temperature": temp,
        "ece": CalibrationMetrics.expected_calibration_error(ts_probs, test_labels),
        "mce": CalibrationMetrics.maximum_calibration_error(ts_probs, test_labels),
        "brier": CalibrationMetrics.brier_score(ts_probs, test_labels),
    }

    # --- Platt Scaling ---
    ps = PlattScaling()
    ps.calibrate(val_logits, val_labels)
    ps_probs = ps.apply(test_logits)
    results["platt_scaling"] = {
        "ece": CalibrationMetrics.expected_calibration_error(ps_probs, test_labels),
        "mce": CalibrationMetrics.maximum_calibration_error(ps_probs, test_labels),
        "brier": CalibrationMetrics.brier_score(ps_probs, test_labels),
    }

    # --- Isotonic Regression ---
    iso = IsotonicRegressionCalibration()
    iso.calibrate(val_logits, val_labels)
    iso_probs = iso.apply(test_logits)
    results["isotonic"] = {
        "ece": CalibrationMetrics.expected_calibration_error(iso_probs, test_labels),
        "mce": CalibrationMetrics.maximum_calibration_error(iso_probs, test_labels),
        "brier": CalibrationMetrics.brier_score(iso_probs, test_labels),
    }

    # --- Reliability diagram comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    method_data = [
        ("Baseline", baseline_probs),
        ("Temperature Scaling", ts_probs),
        ("Platt Scaling", ps_probs),
        ("Isotonic Regression", iso_probs),
    ]
    for ax, (name, probs) in zip(axes.flatten(), method_data):
        _plot_reliability(probs, test_labels, ax, name)
    plt.suptitle(f"Calibration — {model_name} on {dataset_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(cal_dir / f"{model_name}_reliability.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- ECE comparison bar chart ---
    methods = list(results.keys())
    eces = [results[m]["ece"] for m in methods]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, eces, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"],
                   alpha=0.8, edgecolor="black")
    for bar, ece in zip(bars, eces):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{ece:.4f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("ECE")
    plt.title(f"ECE Comparison — {model_name}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(cal_dir / f"{model_name}_ece_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    for m, v in results.items():
        print(f"    {m:25s}  ECE={v['ece']:.4f}  Brier={v['brier']:.4f}")

    return results


def _plot_reliability(probs, labels, ax, title, num_bins=10):
    """Draw a single reliability diagram on *ax*."""
    confs, preds = torch.max(probs, dim=1)
    accs = preds.eq(labels)
    edges = torch.linspace(0, 1, num_bins + 1)
    bin_acc, bin_conf = [], []
    for i in range(num_bins):
        mask = (confs > edges[i]) & (confs <= edges[i + 1])
        if mask.sum() > 0:
            bin_acc.append(accs[mask].float().mean().item())
            bin_conf.append(confs[mask].mean().item())
        else:
            bin_acc.append(0)
            bin_conf.append((edges[i] + edges[i + 1]).item() / 2)
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect")
    ax.bar(bin_conf, bin_acc, width=1 / num_bins, alpha=0.6,
           edgecolor="black", color="steelblue")
    for c, a in zip(bin_conf, bin_acc):
        ax.plot([c, c], [a, c], "r-", alpha=0.5, lw=1.5)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(title); ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)


# ============================================================================
# EXPERIMENT 2: CONFORMAL PREDICTION  (all models)
# ============================================================================

def run_conformal(model_name, dataset_name, val_logits, val_labels,
                  test_logits, test_labels, output_dir, alpha=0.1):
    """Adaptive Prediction Sets (APS) conformal prediction."""
    print(f"\n  [Conformal] {model_name}")
    conf_dir = output_dir / "conformal"
    conf_dir.mkdir(parents=True, exist_ok=True)

    cp = ConformalPrediction(alpha=alpha)
    qhat = cp.calibrate(val_logits, val_labels)
    prediction_sets = cp.predict(test_logits)
    metrics = cp.evaluate_coverage_and_size(prediction_sets, test_labels)

    print(f"    qhat={qhat:.4f}  coverage={metrics['coverage']:.4f}  "
          f"avg_set_size={metrics['avg_set_size']:.2f}")

    # Set-size histogram
    sizes = [len(s) for s in prediction_sets]
    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=range(1, max(sizes) + 2), alpha=0.7, edgecolor="black")
    plt.axvline(np.mean(sizes), color="red", ls="--",
                label=f"Mean: {np.mean(sizes):.2f}")
    plt.xlabel("Prediction Set Size"); plt.ylabel("Count")
    plt.title(f"Conformal Sets — {model_name} on {dataset_name}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(conf_dir / f"{model_name}_set_sizes.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {**metrics, "qhat": qhat}


# ============================================================================
# EXPERIMENT 3: MC DROPOUT  (architecture-gated)
# ============================================================================

def _enable_dropout(model: nn.Module):
    """Turn on Dropout modules while keeping everything else in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def run_mc_dropout(model_name, dataset_name, model, loader, device,
                   output_dir, n_samples=30):
    """MC Dropout uncertainty estimation.

    Skipped automatically for architectures without dropout layers
    (ResNet, DenseNet) — those models have NO stochastic layers to sample.
    """
    if not MC_DROPOUT_ELIGIBLE.get(model_name, False):
        print(f"\n  [MC Dropout] SKIPPED for {model_name}: "
              f"no dropout layers in this architecture")
        return None

    print(f"\n  [MC Dropout] {model_name}  (n_samples={n_samples})")
    mc_dir = output_dir / "mc_dropout"
    mc_dir.mkdir(parents=True, exist_ok=True)

    all_labels = []
    all_mc_probs = []   # shape will be [n_samples, N, C]

    for s in tqdm(range(n_samples), desc="MC samples", leave=False):
        model.eval()
        _enable_dropout(model)   # only Dropout layers → train mode

        sample_probs, sample_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=-1)
                sample_probs.append(probs.cpu())
                if s == 0:
                    sample_labels.append(labels)

        all_mc_probs.append(torch.cat(sample_probs))
        if s == 0:
            all_labels = torch.cat(sample_labels)

    model.eval()  # restore full eval mode

    mc_probs = torch.stack(all_mc_probs)           # [S, N, C]
    mean_probs = mc_probs.mean(dim=0)               # [N, C]
    preds = mean_probs.argmax(dim=1)
    correct = preds.eq(all_labels)

    # Predictive entropy  H[y | x]
    pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

    # Expected entropy  E_theta[ H[y | x, theta] ]  (aleatoric proxy)
    per_sample_entropy = -(mc_probs * torch.log(mc_probs + 1e-10)).sum(dim=-1)  # [S, N]
    expected_entropy = per_sample_entropy.mean(dim=0)  # [N]

    # Mutual information = predictive entropy - expected entropy  (epistemic)
    mutual_info = pred_entropy - expected_entropy

    accuracy = accuracy_score(all_labels.numpy(), preds.numpy())

    results = {
        "accuracy": accuracy,
        "mean_predictive_entropy": pred_entropy.mean().item(),
        "mean_epistemic_uncertainty": mutual_info.mean().item(),
        "mean_entropy_correct": pred_entropy[correct].mean().item() if correct.any() else 0,
        "mean_entropy_incorrect": pred_entropy[~correct].mean().item() if (~correct).any() else 0,
        "mean_epistemic_correct": mutual_info[correct].mean().item() if correct.any() else 0,
        "mean_epistemic_incorrect": mutual_info[~correct].mean().item() if (~correct).any() else 0,
    }

    print(f"    acc={accuracy:.4f}  "
          f"H_pred={results['mean_predictive_entropy']:.4f}  "
          f"MI_epist={results['mean_epistemic_uncertainty']:.4f}")

    # Uncertainty distributions: correct vs incorrect
    if correct.any() and (~correct).any():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist([pred_entropy[correct].numpy(), pred_entropy[~correct].numpy()],
                     label=["Correct", "Incorrect"], bins=30, alpha=0.6)
        axes[0].set_xlabel("Predictive Entropy"); axes[0].set_ylabel("Count")
        axes[0].set_title("Predictive Entropy"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].hist([mutual_info[correct].numpy(), mutual_info[~correct].numpy()],
                     label=["Correct", "Incorrect"], bins=30, alpha=0.6)
        axes[1].set_xlabel("Epistemic Uncertainty (MI)"); axes[1].set_ylabel("Count")
        axes[1].set_title("Epistemic Uncertainty"); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.suptitle(f"MC Dropout — {model_name} on {dataset_name}", fontsize=13)
        plt.tight_layout()
        plt.savefig(mc_dir / f"{model_name}_uncertainty_dist.png",
                    dpi=200, bbox_inches="tight")
        plt.close()

    return results


# ============================================================================
# EXPERIMENT 4: DEEP ENSEMBLES  (all 8 CNN models as ensemble members)
# ============================================================================

def run_deep_ensembles(dataset_name, loader, num_classes, device, output_dir):
    """Treat all 8 trained CNN architectures as an ensemble.

    For each test sample, collect softmax predictions from every model and
    compute ensemble-level uncertainty metrics.
    """
    print(f"\n  [Deep Ensembles] {dataset_name}")
    ens_dir = output_dir / "deep_ensembles"
    ens_dir.mkdir(parents=True, exist_ok=True)

    ensemble_probs = []
    model_names_loaded = []

    for model_name in Config.MODELS:
        try:
            model = load_trained_model(model_name, dataset_name, num_classes, device)
        except FileNotFoundError:
            print(f"    Skipping {model_name}: checkpoint not found")
            continue

        probs_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                probs_list.append(F.softmax(logits, dim=-1).cpu())
                labels_list.append(labels)

        ensemble_probs.append(torch.cat(probs_list))
        model_names_loaded.append(model_name)
        all_labels = torch.cat(labels_list)

        del model
        torch.cuda.empty_cache()

    if len(ensemble_probs) < 2:
        print("    Not enough models for ensembling — skipping")
        return None

    stacked = torch.stack(ensemble_probs)          # [M, N, C]
    mean_probs = stacked.mean(dim=0)                # [N, C]
    preds = mean_probs.argmax(dim=1)
    correct = preds.eq(all_labels)

    # Predictive entropy of ensemble mean
    pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

    # Mean of individual entropies (aleatoric proxy)
    ind_ent = -(stacked * torch.log(stacked + 1e-10)).sum(dim=-1)  # [M, N]
    mean_ind_ent = ind_ent.mean(dim=0)

    # Mutual information (epistemic)
    mutual_info = pred_entropy - mean_ind_ent

    # Jensen-Shannon divergence across members
    jsd = pred_entropy - mean_ind_ent  # equivalent for ensembles

    ensemble_acc = accuracy_score(all_labels.numpy(), preds.numpy())

    # Individual model accuracies
    individual_accs = {}
    for i, mname in enumerate(model_names_loaded):
        m_preds = stacked[i].argmax(dim=1)
        individual_accs[mname] = accuracy_score(all_labels.numpy(), m_preds.numpy())

    results = {
        "ensemble_accuracy": ensemble_acc,
        "individual_accuracies": individual_accs,
        "num_members": len(model_names_loaded),
        "members": model_names_loaded,
        "mean_predictive_entropy": pred_entropy.mean().item(),
        "mean_epistemic_uncertainty": mutual_info.mean().item(),
    }

    print(f"    ensemble_acc={ensemble_acc:.4f}  "
          f"members={len(model_names_loaded)}  "
          f"MI={mutual_info.mean().item():.4f}")
    for mname, acc in individual_accs.items():
        print(f"      {mname:20s} acc={acc:.4f}")

    # Visualization: individual vs ensemble accuracy
    names = list(individual_accs.keys()) + ["ENSEMBLE"]
    accs = list(individual_accs.values()) + [ensemble_acc]
    colors = ["steelblue"] * len(individual_accs) + ["#e74c3c"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, accs, color=colors, alpha=0.8, edgecolor="black")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test Accuracy")
    plt.title(f"Deep Ensemble — {dataset_name}")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(ens_dir / f"{dataset_name}_ensemble_accuracy.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    # Uncertainty distribution
    if correct.any() and (~correct).any():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist([mutual_info[correct].numpy(), mutual_info[~correct].numpy()],
                label=["Correct", "Incorrect"], bins=30, alpha=0.6)
        ax.set_xlabel("Epistemic Uncertainty (MI)")
        ax.set_ylabel("Count")
        ax.set_title(f"Ensemble Epistemic Uncertainty — {dataset_name}")
        ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(ens_dir / f"{dataset_name}_ensemble_uncertainty.png",
                    dpi=200, bbox_inches="tight")
        plt.close()

    return results


# ============================================================================
# EXPERIMENT 5: TEST-TIME AUGMENTATION  (all models)
# ============================================================================

def _get_tta_transforms():
    """Return a list of plausible medical-image augmentations for TTA."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    base = [transforms.Resize((224, 224))]
    augmentations = [
        [],                                              # identity
        [transforms.RandomHorizontalFlip(p=1.0)],
        [transforms.RandomRotation(10)],
        [transforms.RandomAffine(degrees=(-10, -5))],
        [transforms.ColorJitter(brightness=0.2)],
        [transforms.ColorJitter(contrast=0.2)],
        [transforms.RandomAffine(degrees=0, scale=(0.9, 1.0))],
        [transforms.RandomAffine(degrees=0, scale=(1.0, 1.1))],
    ]
    tta_transforms = []
    for aug in augmentations:
        tta_transforms.append(transforms.Compose(base + aug + [
            transforms.ToTensor(), normalize,
        ]))
    return tta_transforms


def run_tta(model_name, dataset_name, model, split_dir, device, output_dir):
    """Test-Time Augmentation uncertainty from prediction variance."""
    print(f"\n  [TTA] {model_name}")
    tta_dir = output_dir / "tta"
    tta_dir.mkdir(parents=True, exist_ok=True)

    tta_transforms = _get_tta_transforms()
    n_aug = len(tta_transforms)

    all_tta_probs = []   # [n_aug, N, C]
    all_labels = None

    for t_idx, tfm in enumerate(tta_transforms):
        ds = MedicalImageDataset(split_dir, "test", tfm)
        loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
        probs_list, labels_list = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                probs_list.append(F.softmax(logits, dim=-1).cpu())
                labels_list.append(labels)
        all_tta_probs.append(torch.cat(probs_list))
        if all_labels is None:
            all_labels = torch.cat(labels_list)

    stacked = torch.stack(all_tta_probs)          # [A, N, C]
    mean_probs = stacked.mean(dim=0)
    preds = mean_probs.argmax(dim=1)
    correct = preds.eq(all_labels)

    pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
    # Variance across augmentations per class, then mean over classes
    tta_variance = stacked.var(dim=0).mean(dim=-1)

    accuracy = accuracy_score(all_labels.numpy(), preds.numpy())

    results = {
        "accuracy": accuracy,
        "n_augmentations": n_aug,
        "mean_entropy": pred_entropy.mean().item(),
        "mean_tta_variance": tta_variance.mean().item(),
    }

    print(f"    acc={accuracy:.4f}  n_aug={n_aug}  "
          f"H={results['mean_entropy']:.4f}  var={results['mean_tta_variance']:.4f}")

    # Uncertainty distribution
    if correct.any() and (~correct).any():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist([pred_entropy[correct].numpy(), pred_entropy[~correct].numpy()],
                     label=["Correct", "Incorrect"], bins=30, alpha=0.6)
        axes[0].set_xlabel("Predictive Entropy"); axes[0].set_ylabel("Count")
        axes[0].set_title("TTA Entropy"); axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].hist([tta_variance[correct].numpy(), tta_variance[~correct].numpy()],
                     label=["Correct", "Incorrect"], bins=30, alpha=0.6)
        axes[1].set_xlabel("TTA Variance"); axes[1].set_ylabel("Count")
        axes[1].set_title("TTA Variance"); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.suptitle(f"TTA — {model_name} on {dataset_name}", fontsize=13)
        plt.tight_layout()
        plt.savefig(tta_dir / f"{model_name}_tta_uncertainty.png",
                    dpi=200, bbox_inches="tight")
        plt.close()

    return results


# ============================================================================
# EXPERIMENT 6: GRAD-CAM  (per-architecture target layers)
# ============================================================================

class GradCAMForCNN:
    """Standard Grad-CAM for convolutional neural networks.

    Targets a specific convolutional layer per architecture (see
    GRADCAM_TARGET_LAYERS registry).  Works with 4-D [B, C, H, W]
    activations — NOT the ViT variant.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """Return a [H, W] heatmap for *target_class*."""
        self.model.eval()
        output = self.model(image)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Global-average-pool gradients over spatial dims → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)     # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True) # [1, 1, h, w]
        cam = F.relu(cam).squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def cleanup(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def run_gradcam(model_name, dataset_name, model, split_dir, device,
                output_dir, num_samples=10):
    """Generate Grad-CAM visualisations using the correct target layer."""
    print(f"\n  [Grad-CAM] {model_name}")
    gc_dir = output_dir / "gradcam" / model_name
    gc_dir.mkdir(parents=True, exist_ok=True)

    target_layer_fn = GRADCAM_TARGET_LAYERS.get(model_name)
    if target_layer_fn is None:
        print(f"    No target layer registered for {model_name} — skipping")
        return None

    target_layer = target_layer_fn(model)
    gradcam = GradCAMForCNN(model, target_layer)

    # Plain transform (no augmentation) for visualization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    vis_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = MedicalImageDataset(split_dir, "test", vis_transform)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)),
                               replace=False)

    for sample_idx in indices:
        img_tensor, label = dataset[sample_idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            logits = model(img_tensor)
            pred_idx = logits.argmax(dim=1).item()
            confidence = F.softmax(logits, dim=-1).max().item()

        # Generate CAM for predicted class
        cam = gradcam.generate_cam(img_tensor, pred_idx)

        # Load original image for overlay
        img_path, _ = dataset.samples[sample_idx]
        orig_img = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig_img.resize((224, 224)))

        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.5 * orig_np + 0.5 * heatmap).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_np); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(cam_resized, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")

        true_cls = idx_to_class.get(label, str(label))
        pred_cls = idx_to_class.get(pred_idx, str(pred_idx))
        status = "correct" if pred_idx == label else "incorrect"
        fig.suptitle(f"True: {true_cls} | Pred: {pred_cls} ({confidence:.3f}) [{status}]",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(gc_dir / f"sample_{sample_idx}.png", dpi=200, bbox_inches="tight")
        plt.close()

    gradcam.cleanup()
    print(f"    Saved {len(indices)} Grad-CAM visualisations → {gc_dir}")
    return {"num_samples": len(indices)}


# ============================================================================
# INCREMENTAL SAVE
# ============================================================================

def _sanitize(obj):
    """Make results JSON-serializable."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    return obj


def _save_results(all_results, results_dir):
    """Write current results to a fixed JSON file (overwritten on each call)."""
    path = os.path.join(results_dir, "cnn_results_latest.json")
    with open(path, "w") as f:
        json.dump(_sanitize(all_results), f, indent=2)
    return path


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CNN Explainability & Uncertainty Experiments",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run a single dataset (e.g. MRI_tumor_binary_norm)")
    parser.add_argument("--model", type=str, default=None,
                        help="Run a single model (e.g. resnet50)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override SPLIT_DIR (e.g. /tmp/split_data for local copy)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override CHECKPOINT_DIR")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        help="Experiments to skip: calibration conformal mc_dropout tta gradcam ensembles")
    parser.add_argument("--mc-samples", type=int, default=30,
                        help="Number of MC Dropout samples (default 30)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    args = parser.parse_args()

    config = Config()
    if args.data_dir:
        config.SPLIT_DIR = args.data_dir
        # Rebuild dataset paths
        for name in config.DATASETS:
            config.DATASETS[name]["path"] = os.path.join(args.data_dir, name)
    if args.checkpoint_dir:
        config.CHECKPOINT_DIR = args.checkpoint_dir
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")
    print(f"Results -> {config.RESULTS_DIR}")

    # Filter datasets / models if specified
    datasets_to_run = config.DATASETS
    if args.dataset:
        if args.dataset not in config.DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(config.DATASETS.keys())}")
            return
        datasets_to_run = {args.dataset: config.DATASETS[args.dataset]}

    models_to_run = config.MODELS
    if args.model:
        if args.model not in config.MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {config.MODELS}")
            return
        models_to_run = [args.model]

    skip = set(args.skip)

    all_results = {}

    for dataset_name, ds_info in datasets_to_run.items():
        ds_path = ds_info["path"]
        if not os.path.exists(ds_path):
            print(f"\nDataset not found: {ds_path} — skipping")
            continue

        output_dir = Path(config.RESULTS_DIR) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 70}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#' * 70}")

        loaders, class_to_idx = get_data_loaders(
            ds_path, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
        )
        num_classes = len(class_to_idx)
        print(f"  Classes ({num_classes}): {list(class_to_idx.keys())}")

        dataset_results = {}

        # ------------------------------------------------------------------
        # Per-model experiments
        # ------------------------------------------------------------------
        for model_name in models_to_run:
            print(f"\n{'=' * 60}")
            print(f"  MODEL: {model_name}")
            print(f"{'=' * 60}")

            try:
                model = load_trained_model(
                    model_name, dataset_name, num_classes, config.DEVICE
                )
            except FileNotFoundError as e:
                print(f"  {e} — skipping")
                continue

            # Collect logits once, reuse everywhere
            val_logits, val_labels = collect_logits(model, loaders["val"], config.DEVICE)
            test_logits, test_labels = collect_logits(model, loaders["test"], config.DEVICE)

            model_results = {}

            # 1. Calibration
            if "calibration" not in skip:
                model_results["calibration"] = run_calibration(
                    model_name, dataset_name, val_logits, val_labels,
                    test_logits, test_labels, output_dir,
                )

            # 2. Conformal Prediction
            if "conformal" not in skip:
                model_results["conformal"] = run_conformal(
                    model_name, dataset_name, val_logits, val_labels,
                    test_logits, test_labels, output_dir,
                )

            # 3. MC Dropout (architecture-gated)
            if "mc_dropout" not in skip:
                model_results["mc_dropout"] = run_mc_dropout(
                    model_name, dataset_name, model, loaders["test"],
                    config.DEVICE, output_dir, n_samples=args.mc_samples,
                )

            # 4. TTA
            if "tta" not in skip:
                model_results["tta"] = run_tta(
                    model_name, dataset_name, model, ds_path,
                    config.DEVICE, output_dir,
                )

            # 5. Grad-CAM
            if "gradcam" not in skip:
                model_results["gradcam"] = run_gradcam(
                    model_name, dataset_name, model, ds_path,
                    config.DEVICE, output_dir,
                )

            dataset_results[model_name] = model_results
            all_results[dataset_name] = dataset_results
            _save_results(all_results, config.RESULTS_DIR)
            print(f"  [saved] intermediate results -> cnn_results_latest.json")

            del model
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # Cross-model: Deep Ensembles
        # ------------------------------------------------------------------
        if "ensembles" not in skip:
            dataset_results["deep_ensembles"] = run_deep_ensembles(
                dataset_name, loaders["test"], num_classes, config.DEVICE, output_dir,
            )
            all_results[dataset_name] = dataset_results
            _save_results(all_results, config.RESULTS_DIR)

    # ------------------------------------------------------------------
    # Final save (timestamped copy + latest)
    # ------------------------------------------------------------------
    latest_path = _save_results(all_results, config.RESULTS_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(config.RESULTS_DIR, f"cnn_results_{ts}.json")
    shutil.copy2(latest_path, final_path)

    print(f"\n{'=' * 70}")
    print(f"ALL CNN EXPERIMENTS COMPLETE")
    print(f"Results saved -> {final_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
