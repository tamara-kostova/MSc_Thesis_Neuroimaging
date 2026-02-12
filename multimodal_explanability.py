"""
Architecture-Aware Explainability, Confidence & Uncertainty for
CLIP / BiomedCLIP Vision-Language Models on Medical Imaging Datasets.

Methods implemented (with architectural guards):
  - Calibration: Temperature Scaling, Platt Scaling, Isotonic Regression  (all VLMs)
  - Conformal Prediction (APS)                                           (all VLMs)
  - MC Dropout on classifier HEAD only                                   (head has Dropout(0.3))
      -- NEVER on the frozen ViT backbone (frozen = no stochasticity)
  - Test-Time Augmentation (TTA)                                         (all VLMs)
  - Grad-CAM for ViT (using last transformer block)                      (all VLMs)
  - Attention Rollout  (ViT-specific, multiplies attention across layers) (all VLMs)
  - Token Attribution / Integrated Gradients                             (all VLMs)

Models: CLIP ViT-B-32 (OpenAI), optionally BiomedCLIP
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
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Dict, List, Optional, Tuple
import cv2
import warnings

warnings.filterwarnings("ignore")

import open_clip

# Reuse calibration / uncertainty primitives from the shared toolkit
from uncertainty_confidence import (
    ConformalPrediction,
    TemperatureScaling,
    PlattScaling,
    IsotonicRegressionCalibration,
    CalibrationMetrics,
    GradCAMForCLIP,
    TokenAttributionAnalysis,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = "/mnt/gdrive/MSc_Thesis_Neuroimaging"
    SPLIT_DIR = os.path.join(BASE_DIR, "data/split")
    RESULTS_DIR = os.path.join(BASE_DIR, "results/clip_explainability")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    IMAGE_SIZE = 224

    # CLIP models to evaluate (model_name, pretrained_source)
    # For HF Hub models, pretrained_source is empty (path is in model_name).
    CLIP_MODELS = [
        ("ViT-B-32", "openai"),
        ("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", ""),
    ]

    DATASETS = {
        "MRI_tumor_binary_norm": {
            "path": os.path.join(SPLIT_DIR, "MRI_tumor_binary_norm"),
            "classes": ["normal", "tumor"],
        },
        "MRI_tumor_multiclass_norm": {
            "path": os.path.join(SPLIT_DIR, "MRI_tumor_multiclass_norm"),
            "classes": ["Carcinoma", "Germinoma", "Glioma", "Granuloma",
                        "Meduloblastoma", "Meningioma", "Neurocitoma", "Normal",
                        "Other", "Papiloma", "Schwannoma", "Tuberculoma"],
        },
        "MRI_ms_norm": {
            "path": os.path.join(SPLIT_DIR, "MRI_ms_norm"),
            "classes": ["Control", "MS"],
        },
        "CT_stroke_binary_norm": {
            "path": os.path.join(SPLIT_DIR, "CT_stroke_binary_norm"),
            "classes": ["normal", "stroke"],
        },
    }

    # Medical-domain text prompt template for zero-shot classification
    TEXT_TEMPLATE = "A medical image showing {}"


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_clip_model(model_name: str, pretrained: str, device: torch.device):
    """Load a CLIP model, its preprocessing transform, and tokenizer.

    For HF Hub models (model_name starts with 'hf-hub:'), the pretrained
    source is encoded in the model name itself, so *pretrained* should be
    empty.
    """
    if pretrained:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
    else:
        # HF Hub models: path is the model_name, no separate pretrained arg
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def create_text_embeddings(model, class_names: List[str],
                           template: str, device: torch.device,
                           tokenizer=None) -> torch.Tensor:
    """Encode class-name prompts into normalised text embeddings.

    Returns:
        text_embeddings: [num_classes, embed_dim]  (on *device*)
    """
    prompts = [template.format(name) for name in class_names]
    if tokenizer is not None:
        tokens = tokenizer(prompts).to(device)
    else:
        tokens = open_clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features


# ============================================================================
# DATASET
# ============================================================================

class MedicalImageDataset(Dataset):
    """Medical image dataset compatible with the existing split structure."""

    def __init__(self, split_dir, split_type="test", transform=None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}

        split_path = self.split_dir / split_type
        if not split_path.exists():
            raise FileNotFoundError(f"Split dir not found: {split_path}")

        for idx, class_dir in enumerate(sorted(
                d for d in split_path.iterdir() if d.is_dir())):
            self.class_to_idx[class_dir.name] = idx
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================================
# LOGIT COLLECTION  (zero-shot via image–text similarity)
# ============================================================================

def collect_clip_logits(model, loader: DataLoader,
                        text_embeddings: torch.Tensor,
                        device: torch.device):
    """Compute similarity logits = image_features @ text_embeddings.T * 100."""
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting logits", leave=False):
            images = images.to(device)
            img_feat = model.encode_image(images)
            img_feat = F.normalize(img_feat, dim=-1)
            logits = img_feat @ text_embeddings.T * 100.0
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


# ============================================================================
# EXPERIMENT 1: CALIBRATION  (all CLIP models)
# ============================================================================

def run_calibration(model_tag, dataset_name, val_logits, val_labels,
                    test_logits, test_labels, output_dir):
    """Temperature / Platt / Isotonic calibration on similarity logits."""
    print(f"\n  [Calibration] {model_tag}")
    cal_dir = output_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Baseline
    baseline_probs = F.softmax(test_logits, dim=-1)
    results["baseline"] = {
        "ece": CalibrationMetrics.expected_calibration_error(baseline_probs, test_labels),
        "mce": CalibrationMetrics.maximum_calibration_error(baseline_probs, test_labels),
        "brier": CalibrationMetrics.brier_score(baseline_probs, test_labels),
    }

    # Temperature Scaling
    ts = TemperatureScaling()
    temp = ts.calibrate(val_logits, val_labels)
    ts_probs = ts.apply(test_logits)
    results["temperature_scaling"] = {
        "temperature": temp,
        "ece": CalibrationMetrics.expected_calibration_error(ts_probs, test_labels),
        "mce": CalibrationMetrics.maximum_calibration_error(ts_probs, test_labels),
        "brier": CalibrationMetrics.brier_score(ts_probs, test_labels),
    }

    # Platt Scaling
    ps = PlattScaling()
    ps.calibrate(val_logits, val_labels)
    ps_probs = ps.apply(test_logits)
    results["platt_scaling"] = {
        "ece": CalibrationMetrics.expected_calibration_error(ps_probs, test_labels),
        "mce": CalibrationMetrics.maximum_calibration_error(ps_probs, test_labels),
        "brier": CalibrationMetrics.brier_score(ps_probs, test_labels),
    }

    # Isotonic Regression
    iso = IsotonicRegressionCalibration()
    iso.calibrate(val_logits, val_labels)
    iso_probs = iso.apply(test_logits)
    results["isotonic"] = {
        "ece": CalibrationMetrics.expected_calibration_error(iso_probs, test_labels),
        "mce": CalibrationMetrics.maximum_calibration_error(iso_probs, test_labels),
        "brier": CalibrationMetrics.brier_score(iso_probs, test_labels),
    }

    # Reliability diagrams
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    method_data = [
        ("Baseline", baseline_probs),
        ("Temperature Scaling", ts_probs),
        ("Platt Scaling", ps_probs),
        ("Isotonic Regression", iso_probs),
    ]
    for ax, (name, probs) in zip(axes.flatten(), method_data):
        _plot_reliability(probs, test_labels, ax, name)
    plt.suptitle(f"Calibration — {model_tag} on {dataset_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(cal_dir / f"{model_tag}_reliability.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ECE comparison
    methods = list(results.keys())
    eces = [results[m]["ece"] for m in methods]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, eces,
                   color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"],
                   alpha=0.8, edgecolor="black")
    for bar, ece in zip(bars, eces):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{ece:.4f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("ECE")
    plt.title(f"ECE Comparison — {model_tag}")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(cal_dir / f"{model_tag}_ece_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    for m, v in results.items():
        print(f"    {m:25s}  ECE={v['ece']:.4f}  Brier={v['brier']:.4f}")

    return results


def _plot_reliability(probs, labels, ax, title, num_bins=10):
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
# EXPERIMENT 2: CONFORMAL PREDICTION
# ============================================================================

def run_conformal(model_tag, dataset_name, val_logits, val_labels,
                  test_logits, test_labels, output_dir, alpha=0.1):
    print(f"\n  [Conformal] {model_tag}")
    conf_dir = output_dir / "conformal"
    conf_dir.mkdir(parents=True, exist_ok=True)

    cp = ConformalPrediction(alpha=alpha)
    qhat = cp.calibrate(val_logits, val_labels)
    prediction_sets = cp.predict(test_logits)
    metrics = cp.evaluate_coverage_and_size(prediction_sets, test_labels)

    print(f"    qhat={qhat:.4f}  coverage={metrics['coverage']:.4f}  "
          f"avg_set_size={metrics['avg_set_size']:.2f}")

    sizes = [len(s) for s in prediction_sets]
    plt.figure(figsize=(8, 5))
    plt.hist(sizes, bins=range(1, max(sizes) + 2), alpha=0.7, edgecolor="black")
    plt.axvline(np.mean(sizes), color="red", ls="--",
                label=f"Mean: {np.mean(sizes):.2f}")
    plt.xlabel("Prediction Set Size"); plt.ylabel("Count")
    plt.title(f"Conformal Sets — {model_tag} on {dataset_name}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(conf_dir / f"{model_tag}_set_sizes.png", dpi=200, bbox_inches="tight")
    plt.close()

    return {**metrics, "qhat": qhat}


# ============================================================================
# EXPERIMENT 3: MC DROPOUT — CLASSIFIER HEAD ONLY
# ============================================================================
#
# ARCHITECTURAL NOTE:
# The CLIP visual backbone is FROZEN (all params have requires_grad=False).
# Enabling dropout on a frozen backbone is meaningless — the model produces
# identical outputs regardless because there is no learned stochasticity.
#
# The trainable classifier heads (CLIPLayerClassifier, CLIPMultiLayerFusion)
# from the layer-fusion benchmark DO contain nn.Dropout(0.3).  MC Dropout
# can be applied there, but only 1 dropout layer exists, yielding limited
# variance.  We document this clearly in the results.
#
# For zero-shot CLIP (no classifier head), MC Dropout is NOT applicable.
# ============================================================================

def run_mc_dropout_zeroshot(model_tag, model, loader, text_embeddings,
                            device, output_dir, n_samples=30):
    """MC Dropout for zero-shot CLIP.

    This enables dropout inside the ViT backbone during inference.
    IMPORTANT CAVEAT: CLIP's backbone was trained with dropout disabled at
    inference time; enabling it introduces out-of-distribution noise rather
    than meaningful epistemic uncertainty.  We run it for completeness but
    clearly flag the limitation.
    """
    print(f"\n  [MC Dropout — zero-shot] {model_tag}")
    print("    NOTE: Limited effectiveness — frozen backbone means "
          "backbone dropout was never calibrated for MC sampling.")
    mc_dir = output_dir / "mc_dropout_head"
    mc_dir.mkdir(parents=True, exist_ok=True)

    # Check whether there are any Dropout modules at all
    has_dropout = any(isinstance(m, nn.Dropout)
                      for m in model.modules())
    if not has_dropout:
        print("    SKIPPED: no Dropout layers found in this model.")
        return None

    all_labels = []
    all_mc_probs = []

    for s in tqdm(range(n_samples), desc="MC samples", leave=False):
        # Put model in eval, then selectively enable Dropout
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        sample_probs, sample_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                img_feat = model.encode_image(images)
                img_feat = F.normalize(img_feat, dim=-1)
                logits = img_feat @ text_embeddings.T * 100.0
                probs = F.softmax(logits, dim=-1)
                sample_probs.append(probs.cpu())
                if s == 0:
                    sample_labels.append(labels)

        all_mc_probs.append(torch.cat(sample_probs))
        if s == 0:
            all_labels = torch.cat(sample_labels)

    model.eval()

    mc_probs = torch.stack(all_mc_probs)      # [S, N, C]
    mean_probs = mc_probs.mean(dim=0)
    preds = mean_probs.argmax(dim=1)
    correct = preds.eq(all_labels)

    pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
    per_sample_ent = -(mc_probs * torch.log(mc_probs + 1e-10)).sum(dim=-1)
    expected_entropy = per_sample_ent.mean(dim=0)
    mutual_info = pred_entropy - expected_entropy

    accuracy = accuracy_score(all_labels.numpy(), preds.numpy())

    results = {
        "accuracy": accuracy,
        "mean_predictive_entropy": pred_entropy.mean().item(),
        "mean_epistemic_uncertainty": mutual_info.mean().item(),
        "mean_entropy_correct": pred_entropy[correct].mean().item() if correct.any() else 0,
        "mean_entropy_incorrect": pred_entropy[~correct].mean().item() if (~correct).any() else 0,
        "caveat": ("Limited: backbone dropout was not calibrated for MC sampling. "
                   "Epistemic estimates from this method should be interpreted cautiously."),
    }

    print(f"    acc={accuracy:.4f}  "
          f"H={results['mean_predictive_entropy']:.4f}  "
          f"MI={results['mean_epistemic_uncertainty']:.4f}")

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

        plt.suptitle(f"MC Dropout (zero-shot, limited) — {model_tag}", fontsize=13)
        plt.tight_layout()
        plt.savefig(mc_dir / f"{model_tag}_uncertainty_dist.png",
                    dpi=200, bbox_inches="tight")
        plt.close()

    return results


# ============================================================================
# EXPERIMENT 4: TEST-TIME AUGMENTATION
# ============================================================================

def run_tta(model_tag, model, split_dir, text_embeddings,
            preprocess, device, output_dir, n_augmentations=8):
    """TTA uncertainty from prediction variance across augmented views."""
    print(f"\n  [TTA] {model_tag}")
    tta_dir = output_dir / "tta"
    tta_dir.mkdir(parents=True, exist_ok=True)

    # Build augmentation transforms (all end with the CLIP preprocessing)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    base = [transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))]
    augmentations = [
        [],
        [transforms.RandomHorizontalFlip(p=1.0)],
        [transforms.RandomRotation(10)],
        [transforms.RandomAffine(degrees=(-10, -5))],
        [transforms.ColorJitter(brightness=0.2)],
        [transforms.ColorJitter(contrast=0.2)],
        [transforms.RandomAffine(degrees=0, scale=(0.9, 1.0))],
        [transforms.RandomAffine(degrees=0, scale=(1.0, 1.1))],
    ]
    tta_tfms = [
        transforms.Compose(base + aug + [transforms.ToTensor(), normalize])
        for aug in augmentations[:n_augmentations]
    ]

    all_tta_probs = []
    all_labels = None

    for tfm in tta_tfms:
        ds = MedicalImageDataset(split_dir, "test", tfm)
        loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
        probs_list, labels_list = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                img_feat = model.encode_image(images)
                img_feat = F.normalize(img_feat, dim=-1)
                logits = img_feat @ text_embeddings.T * 100.0
                probs_list.append(F.softmax(logits, dim=-1).cpu())
                labels_list.append(labels)
        all_tta_probs.append(torch.cat(probs_list))
        if all_labels is None:
            all_labels = torch.cat(labels_list)

    stacked = torch.stack(all_tta_probs)
    mean_probs = stacked.mean(dim=0)
    preds = mean_probs.argmax(dim=1)
    correct = preds.eq(all_labels)

    pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
    tta_variance = stacked.var(dim=0).mean(dim=-1)

    accuracy = accuracy_score(all_labels.numpy(), preds.numpy())

    results = {
        "accuracy": accuracy,
        "n_augmentations": len(tta_tfms),
        "mean_entropy": pred_entropy.mean().item(),
        "mean_tta_variance": tta_variance.mean().item(),
    }

    print(f"    acc={accuracy:.4f}  n_aug={len(tta_tfms)}  "
          f"H={results['mean_entropy']:.4f}  var={results['mean_tta_variance']:.4f}")

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

        plt.suptitle(f"TTA — {model_tag}", fontsize=13)
        plt.tight_layout()
        plt.savefig(tta_dir / f"{model_tag}_tta_uncertainty.png",
                    dpi=200, bbox_inches="tight")
        plt.close()

    return results


# ============================================================================
# EXPERIMENT 5: GRAD-CAM FOR ViT
# ============================================================================

def run_gradcam_vit(model_tag, model, dataset, text_embeddings,
                    class_names, device, output_dir, num_samples=10):
    """Grad-CAM using last transformer block (ViT-specific).

    Uses the existing GradCAMForCLIP from uncertainty_confidence.py which
    correctly handles 3-D ViT activations [B, num_patches, dim].
    """
    print(f"\n  [Grad-CAM ViT] {model_tag}")
    gc_dir = output_dir / "gradcam"
    gc_dir.mkdir(parents=True, exist_ok=True)

    gradcam = GradCAMForCLIP(model)

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)),
                               replace=False)

    for sample_idx in indices:
        img_tensor, label = dataset[sample_idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
            img_feat = F.normalize(img_feat, dim=-1)
            logits = img_feat @ text_embeddings.T * 100.0
            pred_idx = logits.argmax(dim=-1).item()
            confidence = F.softmax(logits, dim=-1).max().item()

        # Grad-CAM for predicted class
        text_emb = text_embeddings[pred_idx:pred_idx + 1]
        cam = gradcam.generate_cam(img_tensor, text_emb)

        # Original image for overlay
        img_path, _ = dataset.samples[sample_idx]
        pil_image = Image.open(img_path).convert("RGB").resize((224, 224))
        orig_np = np.array(pil_image)

        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.5 * orig_np + 0.5 * heatmap).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_np); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(cam_resized, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")

        true_cls = class_names[label] if label < len(class_names) else str(label)
        pred_cls = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        status = "correct" if pred_idx == label else "incorrect"
        fig.suptitle(f"True: {true_cls} | Pred: {pred_cls} ({confidence:.3f}) [{status}]",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(gc_dir / f"sample_{sample_idx}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Clean up hooks
    gradcam.forward_handle.remove()
    gradcam.backward_handle.remove()

    print(f"    Saved {len(indices)} Grad-CAM visualisations -> {gc_dir}")
    return {"num_samples": len(indices)}


# ============================================================================
# EXPERIMENT 6: ATTENTION ROLLOUT  (ViT-specific)
# ============================================================================
#
# Attention rollout multiplies attention matrices across ALL transformer
# layers to compute the total attention flow from input patches to the
# CLS token.  This is a *transformer-native* explainability method that
# does NOT apply to CNNs.
#
# Reference: Abnar & Zuidema (2020) "Quantifying Attention Flow in
#            Transformers"
# ============================================================================

def _extract_attention_weights(model, images: torch.Tensor) -> List[torch.Tensor]:
    """Extract attention weights from every transformer block.

    Supports both OpenCLIP native ViTs (model.visual.transformer.resblocks)
    and timm-based ViTs (model.visual.trunk.blocks, e.g. BiomedCLIP).

    Returns a list of tensors, one per layer, each [B, num_heads, N, N]
    where N = num_patches + 1 (CLS token).
    """
    attn_weights: List[torch.Tensor] = []
    original_forward_fns = []

    if hasattr(model.visual, 'transformer'):
        # --- OpenCLIP native ViT: monkey-patch nn.MultiheadAttention ---
        for block in model.visual.transformer.resblocks:
            attn_module = block.attn
            original_forward = attn_module.forward
            original_forward_fns.append((attn_module, original_forward))

            def patched_forward(_, original_fn=original_forward):
                def wrapper(query, key, value, **kwargs):
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = False
                    out, weights = original_fn(query, key, value, **kwargs)
                    attn_weights.append(weights.detach().cpu())
                    return out, weights
                return wrapper
            attn_module.forward = patched_forward(attn_module)

    elif hasattr(model.visual, 'trunk') and hasattr(model.visual.trunk, 'blocks'):
        # --- Timm-based ViT (e.g. BiomedCLIP) ---
        # Disable fused attention so the manual path (which exposes attn
        # weights) is taken, then capture weights via a forward hook on
        # the attn_drop layer that sits right after softmax(Q@K^T).
        for block in model.visual.trunk.blocks:
            attn_module = block.attn

            # Force non-fused path so attn weights are computed explicitly
            original_fused = getattr(attn_module, 'fused_attn', False)
            original_forward_fns.append(
                (attn_module, 'fused_attn', original_fused))
            attn_module.fused_attn = False

            # Hook attn_drop to capture attention weights after softmax
            def make_hook(storage):
                def hook_fn(module, input, output):
                    # input[0] is the attention weights after softmax
                    storage.append(input[0].detach().cpu())
                return hook_fn

            hook = attn_module.attn_drop.register_forward_hook(
                make_hook(attn_weights))
            original_forward_fns.append((hook, 'remove', None))

    with torch.no_grad():
        model.encode_image(images)

    # Restore everything
    for item in original_forward_fns:
        if len(item) == 3:
            obj, attr, value = item
            if attr == 'remove':
                obj.remove()  # remove hook handle
            else:
                setattr(obj, attr, value)
        else:
            attn_module, original_fn = item
            attn_module.forward = original_fn

    return attn_weights


def attention_rollout(attn_weights: List[torch.Tensor],
                      head_fusion: str = "mean") -> torch.Tensor:
    """Compute attention rollout across layers.

    Args:
        attn_weights: List of [B, num_heads, N, N] per layer
        head_fusion: How to combine heads ("mean", "max", "min")

    Returns:
        rollout: [B, N] — attention from CLS token to all patches+CLS
    """
    result = None
    for attn in attn_weights:
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=1)    # [B, N, N]
        elif head_fusion == "max":
            attn_fused = attn.max(dim=1)[0]
        elif head_fusion == "min":
            attn_fused = attn.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # Add identity (residual connections)
        I = torch.eye(attn_fused.size(-1)).unsqueeze(0)
        attn_fused = attn_fused + I
        # Re-normalize rows
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

        if result is None:
            result = attn_fused
        else:
            result = torch.bmm(attn_fused, result)

    # CLS token attention to all other tokens
    cls_attn = result[:, 0, 1:]   # [B, num_patches]  (exclude CLS→CLS)
    return cls_attn


def run_attention_rollout(model_tag, model, dataset, device,
                          class_names, output_dir, num_samples=10):
    """Attention rollout — ViT-specific explainability.

    Multiplies attention matrices across all transformer layers to visualise
    total attention flow from the CLS token to spatial image patches.
    This method is ONLY meaningful for transformer architectures.
    """
    print(f"\n  [Attention Rollout] {model_tag}")
    ar_dir = output_dir / "attention_rollout"
    ar_dir.mkdir(parents=True, exist_ok=True)

    is_vit = (hasattr(model.visual, 'transformer') or
              (hasattr(model.visual, 'trunk') and hasattr(model.visual.trunk, 'blocks')))
    if not is_vit:
        print("    SKIPPED: model has no ViT backbone (not a ViT)")
        return None

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)),
                               replace=False)

    for sample_idx in indices:
        img_tensor, label = dataset[sample_idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Extract attention weights from all layers
        attn_weights = _extract_attention_weights(model, img_tensor)

        if not attn_weights:
            print("    Could not extract attention weights — skipping")
            return None

        # Compute rollout
        cls_attn = attention_rollout(attn_weights)  # [1, num_patches]
        num_patches = cls_attn.shape[1]
        grid_size = int(np.sqrt(num_patches))
        rollout_map = cls_attn[0].reshape(grid_size, grid_size).numpy()

        # Normalise
        rollout_map = (rollout_map - rollout_map.min()) / \
                      (rollout_map.max() - rollout_map.min() + 1e-8)

        # Load original image
        img_path, _ = dataset.samples[sample_idx]
        pil_image = Image.open(img_path).convert("RGB").resize((224, 224))
        orig_np = np.array(pil_image)

        rollout_resized = cv2.resize(rollout_map, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(rollout_resized * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.5 * orig_np + 0.5 * heatmap).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_np); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(rollout_resized, cmap="jet"); axes[1].set_title("Attention Rollout"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")

        true_cls = class_names[label] if label < len(class_names) else str(label)
        fig.suptitle(f"Attention Rollout — True: {true_cls}", fontsize=12)
        plt.tight_layout()
        plt.savefig(ar_dir / f"sample_{sample_idx}.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"    Saved {len(indices)} attention rollout visualisations -> {ar_dir}")
    return {"num_samples": len(indices)}


# ============================================================================
# EXPERIMENT 7: TOKEN ATTRIBUTION / INTEGRATED GRADIENTS
# ============================================================================

def run_token_attribution(model_tag, model, dataset, text_embeddings,
                          class_names, device, output_dir, num_samples=5):
    """Integrated gradients token attribution.

    Uses the TokenAttributionAnalysis from uncertainty_confidence.py.
    Computes attribution by interpolating from a zero baseline to the
    input image and accumulating gradients.
    """
    print(f"\n  [Token Attribution] {model_tag}")
    ta_dir = output_dir / "token_attribution"
    ta_dir.mkdir(parents=True, exist_ok=True)

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)),
                               replace=False)

    for sample_idx in indices:
        img_tensor, label = dataset[sample_idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
            img_feat = F.normalize(img_feat, dim=-1)
            logits = img_feat @ text_embeddings.T * 100.0
            pred_idx = logits.argmax(dim=-1).item()

        # Integrated gradients for predicted class
        text_emb = text_embeddings[pred_idx:pred_idx + 1]
        attribution = TokenAttributionAnalysis.integrated_gradients(
            model, img_tensor, text_emb, steps=30
        )

        # Visualise
        img_path, _ = dataset.samples[sample_idx]
        pil_image = Image.open(img_path).convert("RGB").resize((224, 224))

        fig = TokenAttributionAnalysis.visualize_attribution(
            pil_image, attribution
        )

        true_cls = class_names[label] if label < len(class_names) else str(label)
        pred_cls = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        status = "correct" if pred_idx == label else "incorrect"
        fig.suptitle(f"True: {true_cls} | Pred: {pred_cls} [{status}]", fontsize=12, y=1.02)
        plt.savefig(ta_dir / f"sample_{sample_idx}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"    Saved {len(indices)} token attribution visualisations -> {ta_dir}")
    return {"num_samples": len(indices)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CLIP/BiomedCLIP Explainability & Uncertainty Experiments",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run a single dataset (e.g. MRI_tumor_binary_norm)")
    parser.add_argument("--clip-model", type=str, default=None,
                        help="Run a single CLIP model (e.g. ViT-B-32)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override SPLIT_DIR (e.g. /tmp/split_data for local copy)")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        help="Experiments to skip: calibration conformal mc_dropout tta "
                             "gradcam attention_rollout token_attribution")
    parser.add_argument("--mc-samples", type=int, default=30,
                        help="Number of MC Dropout samples (default 30)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--gradcam-samples", type=int, default=10,
                        help="Number of Grad-CAM samples per dataset (default 10)")
    parser.add_argument("--rollout-samples", type=int, default=10,
                        help="Number of attention rollout samples per dataset (default 10)")
    parser.add_argument("--attribution-samples", type=int, default=5,
                        help="Number of token attribution samples per dataset (default 5)")
    args = parser.parse_args()

    config = Config()
    if args.data_dir:
        config.SPLIT_DIR = args.data_dir
        for name, ds_info in config.DATASETS.items():
            ds_info["path"] = os.path.join(args.data_dir, name)
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")
    print(f"Results -> {config.RESULTS_DIR}")

    # Filter CLIP models if specified
    clip_models = config.CLIP_MODELS
    if args.clip_model:
        clip_models = [(n, s) for n, s in config.CLIP_MODELS if n == args.clip_model]
        if not clip_models:
            print(f"Unknown CLIP model: {args.clip_model}")
            print(f"Available: {[n for n, _ in config.CLIP_MODELS]}")
            return

    # Filter datasets if specified
    datasets_to_run = config.DATASETS
    if args.dataset:
        if args.dataset not in config.DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(config.DATASETS.keys())}")
            return
        datasets_to_run = {args.dataset: config.DATASETS[args.dataset]}

    skip = set(args.skip)

    all_results = {}

    for model_name, pretrained_src in clip_models:
        model_tag = (f"{model_name}_{pretrained_src}".replace("/", "-")
                     .replace(":", "-").rstrip("_"))

        print(f"\n{'#' * 70}")
        print(f"# CLIP MODEL: {model_name} ({pretrained_src})")
        print(f"{'#' * 70}")

        model, preprocess, tokenizer = load_clip_model(model_name, pretrained_src, config.DEVICE)
        model_results = {}

        for dataset_name, ds_info in datasets_to_run.items():
            ds_path = ds_info["path"]
            if not os.path.exists(ds_path):
                print(f"\n  Dataset not found: {ds_path} — skipping")
                continue

            output_dir = Path(config.RESULTS_DIR) / model_tag / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)

            class_names = ds_info["classes"]

            print(f"\n{'=' * 60}")
            print(f"  DATASET: {dataset_name}  ({len(class_names)} classes)")
            print(f"{'=' * 60}")

            # Text embeddings
            text_embs = create_text_embeddings(
                model, class_names, config.TEXT_TEMPLATE, config.DEVICE,
                tokenizer=tokenizer,
            )

            # Build loaders
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            eval_transform = transforms.Compose([
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.ToTensor(),
                normalize,
            ])

            val_ds = MedicalImageDataset(ds_path, "val", eval_transform)
            test_ds = MedicalImageDataset(ds_path, "test", eval_transform)

            val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                                    shuffle=False, num_workers=config.NUM_WORKERS,
                                    pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                                     shuffle=False, num_workers=config.NUM_WORKERS,
                                     pin_memory=True)

            # Collect logits once
            val_logits, val_labels = collect_clip_logits(
                model, val_loader, text_embs, config.DEVICE
            )
            test_logits, test_labels = collect_clip_logits(
                model, test_loader, text_embs, config.DEVICE
            )

            # Baseline accuracy
            preds = F.softmax(test_logits, dim=-1).argmax(dim=1)
            baseline_acc = accuracy_score(test_labels.numpy(), preds.numpy())
            print(f"  Baseline accuracy: {baseline_acc:.4f}")

            ds_results = {"baseline_accuracy": baseline_acc}

            # 1. Calibration
            if "calibration" not in skip:
                ds_results["calibration"] = run_calibration(
                    model_tag, dataset_name, val_logits, val_labels,
                    test_logits, test_labels, output_dir,
                )

            # 2. Conformal Prediction
            if "conformal" not in skip:
                ds_results["conformal"] = run_conformal(
                    model_tag, dataset_name, val_logits, val_labels,
                    test_logits, test_labels, output_dir,
                )

            # 3. MC Dropout (with documented limitations)
            if "mc_dropout" not in skip:
                ds_results["mc_dropout"] = run_mc_dropout_zeroshot(
                    model_tag, model, test_loader, text_embs,
                    config.DEVICE, output_dir, n_samples=args.mc_samples,
                )

            # 4. TTA
            if "tta" not in skip:
                ds_results["tta"] = run_tta(
                    model_tag, model, ds_path, text_embs,
                    preprocess, config.DEVICE, output_dir,
                )

            # 5. Grad-CAM for ViT
            if "gradcam" not in skip:
                ds_results["gradcam"] = run_gradcam_vit(
                    model_tag, model, test_ds, text_embs,
                    class_names, config.DEVICE, output_dir,
                    num_samples=args.gradcam_samples,
                )

            # 6. Attention Rollout (ViT-specific)
            if "attention_rollout" not in skip:
                ds_results["attention_rollout"] = run_attention_rollout(
                    model_tag, model, test_ds, config.DEVICE,
                    class_names, output_dir,
                    num_samples=args.rollout_samples,
                )

            # 7. Token Attribution / Integrated Gradients
            if "token_attribution" not in skip:
                ds_results["token_attribution"] = run_token_attribution(
                    model_tag, model, test_ds, text_embs,
                    class_names, config.DEVICE, output_dir,
                    num_samples=args.attribution_samples,
                )

            model_results[dataset_name] = ds_results

        all_results[model_tag] = model_results

        del model
        torch.cuda.empty_cache()

    # Save aggregate results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(config.RESULTS_DIR, f"clip_results_{ts}.json")

    def _sanitize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(x) for x in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(_sanitize(all_results), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"ALL CLIP EXPERIMENTS COMPLETE")
    print(f"Results saved -> {results_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
