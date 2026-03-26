#!/usr/bin/env python3
"""
16_multimodal_clip_classification.py

Zero-shot and linear-probe classification using OpenAI CLIP and BioMedCLIP
on the four neuroimaging datasets.

Usage:
    python 16_multimodal_clip_classification.py               # all methods
    python 16_multimodal_clip_classification.py --methods zeroshot
    python 16_multimodal_clip_classification.py --methods linearprobe
    python 16_multimodal_clip_classification.py --datasets MRI_tumor_binary_norm MRI_ms_norm

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
import seaborn as sns
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

    def __init__(self):
        os.makedirs(self.RESULTS_DIR,    exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)


config = Config()
print(f"Results dir: {config.RESULTS_DIR}")
print(f"Device:      {config.DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# TEXT PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

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
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    models = {}

    print("Loading OpenAI CLIP (ViT-B/32)...")
    clip_model, _, clip_pre = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(config.DEVICE)
    clip_tok   = open_clip.get_tokenizer("ViT-B-32")
    models["OpenAI-CLIP"] = {
        "model": clip_model, "tokenizer": clip_tok, "preprocess": clip_pre,
        "embed_dim": 512,
    }
    print("  OpenAI CLIP loaded")

    print("Loading BioMedCLIP...")
    try:
        bm_id  = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        bm_m, _, bm_pre = open_clip.create_model_and_transforms(bm_id)
        bm_m   = bm_m.to(config.DEVICE)
        bm_tok = open_clip.get_tokenizer(bm_id)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(config.DEVICE)
            bm_dim = bm_m.encode_image(dummy).shape[1]
        models["BioMedCLIP"] = {
            "model": bm_m, "tokenizer": bm_tok, "preprocess": bm_pre,
            "embed_dim": bm_dim,
        }
        print("  BioMedCLIP loaded")
    except Exception as e:
        print(f"  BioMedCLIP not available: {e}")

    return models

# ─────────────────────────────────────────────────────────────────────────────
# ZERO-SHOT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def encode_text_prompts(model, tokenizer, class_prompts: dict) -> dict:
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for cls, prompt in class_prompts.items():
            tokens = tokenizer([prompt]).to(config.DEVICE)
            emb    = model.encode_text(tokens)
            embeddings[cls] = F.normalize(emb, dim=-1)
    return embeddings


def zero_shot_eval(model_info: dict, test_loader, dataset_name: str,
                   model_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Zero-Shot: {model_name} on {dataset_name}")
    print(f"{'='*60}")

    model     = model_info["model"]
    tokenizer = model_info["tokenizer"]
    prompts   = CLASS_PROMPTS[dataset_name]
    text_embs = encode_text_prompts(model, tokenizer, prompts)

    class_names     = list(prompts.keys())
    text_emb_matrix = torch.stack(
        [text_embs[cn] for cn in class_names]).squeeze(1)  # [C, D]

    all_preds, all_labels, all_probs = [], [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Zero-shot eval"):
            images = batch["image"].to(config.DEVICE)
            labels = batch["class_idx"]

            img_emb = F.normalize(model.encode_image(images), dim=-1)
            logits  = (img_emb @ text_emb_matrix.T) * 100
            probs   = F.softmax(logits, dim=1)
            preds   = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return _compute_metrics(all_preds, all_labels, all_probs,
                            model_name, dataset_name, "zero-shot", class_names)

# ─────────────────────────────────────────────────────────────────────────────
# LINEAR PROBE
# ─────────────────────────────────────────────────────────────────────────────

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

    classifier = LinearClassifier(model_info["embed_dim"], num_classes).to(config.DEVICE)
    optimizer  = optim.AdamW(classifier.parameters(),
                             lr=config.LEARNING_RATE_LINEAR,
                             weight_decay=config.WEIGHT_DECAY)
    criterion  = nn.CrossEntropyLoss()
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc    = 0.0
    patience_counter = 0
    best_state      = None

    for epoch in range(config.NUM_EPOCHS_LINEAR):
        # Train
        classifier.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS_LINEAR} [Train]"):
            images = batch["image"].to(config.DEVICE)
            labels = batch["class_idx"].to(config.DEVICE)
            with torch.no_grad():
                features = F.normalize(model.encode_image(images), dim=-1)
            outputs = classifier(features)
            loss    = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            t_correct += outputs.argmax(1).eq(labels).sum().item()
            t_total   += labels.size(0)

        t_loss /= len(train_loader)
        t_acc   = t_correct / t_total

        # Val
        classifier.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader,
                              desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS_LINEAR} [Val]"):
                images  = batch["image"].to(config.DEVICE)
                labels  = batch["class_idx"].to(config.DEVICE)
                features = F.normalize(model.encode_image(images), dim=-1)
                outputs  = classifier(features)
                v_loss  += criterion(outputs, labels).item()
                v_correct += outputs.argmax(1).eq(labels).sum().item()
                v_total   += labels.size(0)

        v_loss /= len(val_loader)
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

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Linear probe test"):
            images  = batch["image"].to(config.DEVICE)
            labels  = batch["class_idx"]
            features = F.normalize(model.encode_image(images), dim=-1)
            outputs  = classifier(features)
            probs    = F.softmax(outputs, dim=1)
            preds    = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    num_classes = np.array(all_probs).shape[1]
    class_names = [str(i) for i in range(num_classes)]
    return _compute_metrics(all_preds, all_labels, all_probs,
                            model_name, dataset_name, "linear-probe", class_names)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(preds, labels, probs, model_name, dataset_name,
                     method, class_names) -> dict:
    preds  = np.array(preds)
    labels = np.array(labels)
    probs  = np.array(probs)
    n      = len(class_names)

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

    print(f"  Accuracy: {accuracy:.4f}  F1: {f1:.4f}  "
          f"AUC: {auc:.4f}" if auc else f"  Accuracy: {accuracy:.4f}  F1: {f1:.4f}  AUC: N/A")

    return {
        "model": model_name, "dataset": dataset_name, "method": method,
        "accuracy": float(accuracy), "f1": float(f1),
        "auc": float(auc) if auc is not None else None,
        "num_classes": n,
    }

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, filename: str, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["accuracy", "f1", "auc"]):
        try:
            pivot = df.pivot(index="dataset", columns="model", values=metric)
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"{metric.upper()} by Dataset",
                         fontsize=13, fontweight="bold")
            ax.set_ylabel(metric.capitalize())
            ax.set_xlabel("")
            ax.legend(title="Model")
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
        except Exception as e:
            ax.set_title(f"{metric} — {e}")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {path}")
    plt.show()


def print_summary(df: pd.DataFrame, label: str):
    print(f"\n{'='*70}")
    print(f"{label} — SUMMARY")
    print(f"{'='*70}")
    print(df.to_string(index=False))

    avg = df.groupby("model")[["accuracy", "f1", "auc"]].mean()
    print("\nAverage by model:")
    print(avg.to_string())

    best = df.loc[df.groupby("dataset")["accuracy"].idxmax()]
    print("\nBest model per dataset:")
    print(best[["dataset", "model", "accuracy", "f1"]].to_string(index=False))


def save_report(df_zs: pd.DataFrame, df_lp: pd.DataFrame):
    combined = pd.concat([df_zs, df_lp], ignore_index=True)
    combined.to_csv(os.path.join(config.RESULTS_DIR, "all_results.csv"), index=False)

    avg = combined.groupby("model")[["accuracy", "f1", "auc"]].mean()
    report = {
        "experiment":  "Multimodal Neuroimaging Classification — CLIP",
        "date":        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models":      combined["model"].unique().tolist(),
        "datasets":    combined["dataset"].unique().tolist(),
        "methods":     combined["method"].unique().tolist(),
        "results":     combined.to_dict("records"),
        "summary": {
            "avg_accuracy_by_model":  avg["accuracy"].to_dict(),
            "avg_f1_by_model":        avg["f1"].to_dict(),
            "best_overall_model":     avg["accuracy"].idxmax(),
            "best_overall_accuracy":  float(avg["accuracy"].max()),
        },
    }
    report_path = os.path.join(config.RESULTS_DIR, "experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")

    md_path = os.path.join(config.RESULTS_DIR, "results_summary.md")
    with open(md_path, "w") as f:
        f.write("# Multimodal CLIP Results\n\n")
        f.write(f"**Date:** {report['date']}\n\n")
        f.write("## Results\n\n")
        f.write(combined.to_markdown(index=False))
        f.write(f"\n\n**Best model:** {report['summary']['best_overall_model']} "
                f"({report['summary']['best_overall_accuracy']:.4f})\n")
    print(f"Markdown saved: {md_path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(datasets: list[str], methods: list[str]):
    all_models = load_models()

    zs_results = []
    lp_results = []

    for dataset in datasets:
        print(f"\n\n{'#'*70}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*70}")

        # ── Zero-shot ───────────────────────────────────────────────────────
        if "zeroshot" in methods:
            for model_name, model_info in all_models.items():
                test_ds = MedicalImageDataset(
                    dataset, "test", model_info["preprocess"])
                test_dl = DataLoader(
                    test_ds, batch_size=config.BATCH_SIZE,
                    shuffle=False, num_workers=2)
                result = zero_shot_eval(model_info, test_dl, dataset, model_name)
                zs_results.append(result)

        # ── Linear probe (OpenAI CLIP only) ─────────────────────────────────
        if "linearprobe" in methods:
            model_info  = all_models["OpenAI-CLIP"]
            pre         = model_info["preprocess"]
            num_classes = len(CLASS_PROMPTS[dataset])

            train_dl = DataLoader(
                MedicalImageDataset(dataset, "train", pre),
                batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
            val_dl = DataLoader(
                MedicalImageDataset(dataset, "val", pre),
                batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
            test_dl = DataLoader(
                MedicalImageDataset(dataset, "test", pre),
                batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

            classifier, _ = linear_probe_train(
                model_info, train_dl, val_dl, num_classes,
                "OpenAI-CLIP", dataset)
            result = linear_probe_eval(
                model_info, classifier, test_dl, "OpenAI-CLIP", dataset)
            lp_results.append(result)

    # ── Results ─────────────────────────────────────────────────────────────
    df_zs = pd.DataFrame(zs_results)
    df_lp = pd.DataFrame(lp_results)

    if not df_zs.empty:
        df_zs.to_csv(os.path.join(config.RESULTS_DIR, "zero_shot_results.csv"),
                     index=False)
        print_summary(df_zs, "Zero-Shot")
        plot_results(df_zs, "zero_shot_comparison.png", "Zero-Shot CLIP Results")

    if not df_lp.empty:
        df_lp.to_csv(os.path.join(config.RESULTS_DIR, "linear_probe_results.csv"),
                     index=False)
        print_summary(df_lp, "Linear Probe")
        plot_results(df_lp, "linear_probe_comparison.png", "Linear Probe Results")

    if not df_zs.empty or not df_lp.empty:
        save_report(df_zs, df_lp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=config.DATASETS,
                        help="Datasets to evaluate")
    parser.add_argument("--methods",  nargs="+",
                        choices=["zeroshot", "linearprobe"],
                        default=["zeroshot", "linearprobe"],
                        help="Methods to run")
    args = parser.parse_args()

    main(datasets=args.datasets, methods=args.methods)
