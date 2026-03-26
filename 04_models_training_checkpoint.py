#!/usr/bin/env python3
"""
04_models_training_checkpoint.py

Benchmark training of multiple CNN architectures (ResNet50/101, VGG16,
DenseNet121/169, MobileNetV2, EfficientNet-B0/B4) across four neuroimaging
datasets (MRI tumour binary/multiclass, MS, CT stroke).

Supports checkpointing and resume across sessions via ProgressTracker.
Cells 15-21 (initialisation, training loops, results, visualisations) are
executed only when the script is run directly.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from pathlib import Path
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')


class Config:
    """Global configuration for all models"""

    # Paths
    BASE_DIR = os.environ.get("THESIS_DIR", os.path.expanduser("~/Documents/MSc_Thesis_Neuroimaging"))
    SPLIT_DIR = f"{BASE_DIR}/data/split"
    RESULTS_DIR = f"{BASE_DIR}/results/benchmarks"
    CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints"

    # NEW: Progress tracking
    PROGRESS_FILE = f"{CHECKPOINT_DIR}/training_progress.json"

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random seed
    SEED = 42

    # Datasets to train on
    DATASETS = [
        "MRI_tumor_binary_norm",
        "MRI_tumor_multiclass_norm",
        "MRI_ms_norm",
        "CT_stroke_binary_norm"
    ]

    # Models to benchmark
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

    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 1e-3

    # NEW: Checkpoint saving frequency (save every N epochs)
    CHECKPOINT_FREQ = 2

    def __init__(self):
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)


class ProgressTracker:
    """Track training progress across Colab sessions"""

    def __init__(self, progress_file):
        self.progress_file = progress_file
        self.progress = self.load_progress()

    def load_progress(self):
        """Load existing progress or create new"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'completed': [],
            'in_progress': None,
            'last_epoch': 0,
            'results': {}
        }

    def save_progress(self):
        """Save progress to disk"""
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def is_completed(self, dataset, model):
        """Check if dataset-model combination is already done"""
        return [dataset, model] in self.progress['completed']

    def mark_started(self, dataset, model):
        """Mark a dataset-model pair as started"""
        self.progress['in_progress'] = [dataset, model]
        self.progress['last_epoch'] = 0
        self.save_progress()

    def update_epoch(self, epoch):
        """Update last completed epoch"""
        self.progress['last_epoch'] = epoch
        self.save_progress()

    def mark_completed(self, dataset, model, results):
        """Mark a dataset-model pair as completed"""
        self.progress['completed'].append([dataset, model])

        if dataset not in self.progress['results']:
            self.progress['results'][dataset] = {}
        self.progress['results'][dataset][model] = results

        self.progress['in_progress'] = None
        self.progress['last_epoch'] = 0
        self.save_progress()

    def get_resume_info(self, dataset, model):
        """Get info to resume training"""
        if (self.progress['in_progress'] == [dataset, model] and
            self.progress['last_epoch'] > 0):
            return self.progress['last_epoch']
        return 0

    def get_results(self):
        """Get all completed results"""
        return self.progress['results']

    def print_status(self):
        """Print current progress status"""
        print("\n" + "="*70)
        print("TRAINING PROGRESS STATUS")
        print("="*70)
        print(f"Completed: {len(self.progress['completed'])} dataset-model pairs")

        if self.progress['completed']:
            print("\nCompleted pairs:")
            for dataset, model in self.progress['completed']:
                print(f"  ✓ {dataset} - {model}")

        if self.progress['in_progress']:
            dataset, model = self.progress['in_progress']
            print(f"\nIn Progress: {dataset} - {model}")
            print(f"  Last epoch: {self.progress['last_epoch']}")

        print("="*70 + "\n")


class MedicalImageDataset(Dataset):
    """PyTorch Dataset for medical images with stratified splits"""

    def __init__(self, split_dir, split_type="train", transform=None):
        """
        Args:
            split_dir: path to split directory
            split_type: "train", "val", or "test"
            transform: image transformations
        """
        self.split_dir = split_dir
        self.split_type = split_type
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}
        self._build_samples()

    def _build_samples(self):
        """Build list of (path, label) tuples"""
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
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        img_path, label = self.samples[idx]

        # Load as grayscale and convert to RGB (3 channels for pretrained models)
        image = Image.open(img_path).convert('L')
        image_rgb = Image.new('RGB', image.size)
        image_rgb.paste(image)

        if self.transform:
            image_rgb = self.transform(image_rgb)

        return image_rgb, label


def get_data_loaders(split_dir, batch_size=32, num_workers=2):
    """Create train/val/test DataLoaders"""

    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=5, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        normalize,
    ])

    # Val/Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # Create datasets
    train_ds = MedicalImageDataset(split_dir, "train", train_transform)
    val_ds = MedicalImageDataset(split_dir, "val", test_transform)
    test_ds = MedicalImageDataset(split_dir, "test", test_transform)

    # Create loaders
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
    }

    return loaders, train_ds.class_to_idx


def create_model(model_name, num_classes, pretrained=True):
    """Create model with specified architecture"""

    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet101":
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "densenet169":
        model = models.densenet169(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

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


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_epoch = None
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model.state_dict().copy()
            self.best_epoch = 0
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict().copy()
            self.best_epoch = 0
        else:
            self.counter += 1
            self.best_epoch += 1

        return self.counter >= self.patience

    def restore_best_weights(self, model):
        if self.best_state is not None and self.restore_best:
            model.load_state_dict(self.best_state)

    def state_dict(self):
        """Return state for checkpointing"""
        return {
            'counter': self.counter,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'best_state': self.best_state
        }

    def load_state_dict(self, state):
        """Load state from checkpoint"""
        self.counter = state['counter']
        self.best_loss = state['best_loss']
        self.best_epoch = state['best_epoch']
        self.best_state = state['best_state']


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        with torch.no_grad():
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

        pbar.update(1)

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_acc = accuracy_score(all_labels, all_preds)

    return avg_loss, avg_acc


def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

            pbar.update(1)

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    avg_acc = accuracy_score(all_labels, all_preds)

    # Compute AUC
    try:
        if len(np.unique(all_labels)) == 2:
            avg_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            avg_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception as e:
        print(f"  Warning: Could not compute AUC: {e}")
        avg_auc = 0.0

    return avg_loss, avg_acc, avg_auc


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, early_stop,
                   epoch, history):
    """Save complete checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'early_stop_state': early_stop.state_dict(),
        'history': history,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, early_stop):
    """Load checkpoint and return start epoch and history"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    early_stop.load_state_dict(checkpoint['early_stop_state'])

    return checkpoint['epoch'] + 1, checkpoint['history']


def train_model(model, loaders, criterion, optimizer, scheduler, device,
                num_epochs, model_name, dataset_name, checkpoint_dir,
                progress_tracker, start_epoch=0, resume_history=None):
    """Train model with checkpointing and resume capability"""

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}.pt")
    early_stop = EarlyStopping(patience=Config.PATIENCE, min_delta=Config.MIN_DELTA)

    # Resume from checkpoint if available
    if start_epoch > 0 and os.path.exists(checkpoint_path):
        print(f"\n🔄 RESUMING from epoch {start_epoch}")
        try:
            start_epoch, history = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, early_stop
            )
            model = model.to(device)
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("Starting from scratch...")
            start_epoch = 0
            history = {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [], 'val_auc': []
            }
    else:
        history = resume_history if resume_history else {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }

    print(f"\n{'='*70}")
    print(f"Training {model_name} on {dataset_name}")
    print(f"Epochs: {start_epoch} → {num_epochs}")
    print(f"{'='*70}")

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_epoch(
            model, loaders['train'], criterion, optimizer, device
        )

        val_loss, val_acc, val_auc = validate_epoch(
            model, loaders['val'], criterion, device
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"TrLoss: {train_loss:.4f} | TrAcc: {train_acc:.4f} | "
              f"VaLoss: {val_loss:.4f} | VaAcc: {val_acc:.4f} | VaAUC: {val_auc:.4f}")

        if scheduler is not None:
            scheduler.step(val_loss)

        # Save checkpoint every N epochs
        if (epoch + 1) % Config.CHECKPOINT_FREQ == 0:
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler,
                early_stop, epoch, history
            )
            progress_tracker.update_epoch(epoch)
            print(f"  💾 Checkpoint saved (epoch {epoch+1})")

        if early_stop(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            early_stop.restore_best_weights(model)
            # Save final checkpoint
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler,
                early_stop, epoch, history
            )
            break

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}_final.pt")
    torch.save(model.state_dict(), final_model_path)

    return history, final_model_path


def evaluate_model(model, loader, device):
    """Full evaluation metrics"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
    }

    # AUC
    try:
        if len(np.unique(all_labels)) == 2:
            metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception as e:
        print(f"  Warning: Could not compute AUC: {e}")
        metrics['auc'] = 0.0

    return metrics, all_preds, all_labels


def save_results(results, output_path):
    """Save results to JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def analyze_convergence(history, threshold=0.001):
    """
    Determine when a model converged based on validation loss stabilization

    Args:
        history: dict with 'val_loss' key
        threshold: max change in loss to consider converged

    Returns:
        (convergence_epoch, convergence_loss)
    """
    val_loss = history['val_loss']

    if len(val_loss) < 3:
        return len(val_loss), min(val_loss)

    # Find first epoch where loss change is below threshold for 3 consecutive epochs
    for i in range(2, len(val_loss)):
        if i >= len(val_loss) - 1:
            break

        # Check if recent changes are small
        recent_changes = [abs(val_loss[j] - val_loss[j-1])
                         for j in range(i-1, min(i+2, len(val_loss)))]

        if all(change < threshold for change in recent_changes):
            return i, val_loss[i]

    return len(val_loss), min(val_loss)


def calculate_learning_speed(history):
    """Calculate how fast the model learns (loss reduction per epoch)"""
    val_loss = history['val_loss']

    if len(val_loss) < 2:
        return 0

    # Average loss reduction in first 5 epochs
    early_epochs = min(5, len(val_loss))
    early_reduction = (val_loss[0] - val_loss[early_epochs-1]) / early_epochs

    return early_reduction


def calculate_overfitting_gap(history):
    """Calculate train-val performance gap"""
    if not history['train_acc'] or not history['val_acc']:
        return 0, 0

    # Average gap in last 3 epochs
    last_n = min(3, len(history['train_acc']))

    acc_gap = np.mean(history['train_acc'][-last_n:]) - np.mean(history['val_acc'][-last_n:])
    loss_gap = np.mean(history['val_loss'][-last_n:]) - np.mean(history['train_loss'][-last_n:])

    return acc_gap, loss_gap


def load_and_process_data(json_path):
    """Load JSON and create analysis dataframes"""

    print(f"Loading data from: {json_path}")
    with open(json_path, 'r') as f:
        results = json.load(f)

    print(f"Found {len(results)} datasets")

    # 1. Convergence data
    convergence_data = []
    for dataset_name, models in results.items():
        for model_name, model_data in models.items():
            if 'history' not in model_data:
                continue

            history = model_data['history']
            conv_epoch, conv_loss = analyze_convergence(history)

            convergence_data.append({
                'dataset': dataset_name,
                'model': model_name,
                'convergence_epoch': conv_epoch,
                'convergence_val_loss': conv_loss,
                'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
                'final_val_auc': history['val_auc'][-1] if history['val_auc'] else 0,
                'total_epochs': len(history['train_loss']),
                'best_val_loss': min(history['val_loss']) if history['val_loss'] else 1.0,
                'params': model_data.get('params', 0)
            })

    conv_df = pd.DataFrame(convergence_data)

    # 2. Learning speed data
    learning_speed_data = []
    for dataset_name, models in results.items():
        for model_name, model_data in models.items():
            if 'history' not in model_data:
                continue

            history = model_data['history']
            speed = calculate_learning_speed(history)

            learning_speed_data.append({
                'dataset': dataset_name,
                'model': model_name,
                'learning_speed': speed,
                'epoch_1_val_acc': history['val_acc'][0] if history['val_acc'] else 0,
                'epoch_1_val_loss': history['val_loss'][0] if history['val_loss'] else 1.0,
            })

    speed_df = pd.DataFrame(learning_speed_data)

    # 3. Overfitting data
    overfitting_data = []
    for dataset_name, models in results.items():
        for model_name, model_data in models.items():
            if 'history' not in model_data:
                continue

            history = model_data['history']
            acc_gap, loss_gap = calculate_overfitting_gap(history)

            overfitting_data.append({
                'dataset': dataset_name,
                'model': model_name,
                'acc_gap': acc_gap,
                'loss_gap': loss_gap,
                'overfitting_score': (acc_gap + loss_gap) / 2
            })

    overfit_df = pd.DataFrame(overfitting_data)

    print(f"Processed {len(conv_df)} model-dataset combinations")

    return results, conv_df, speed_df, overfit_df


# ============================================================================
# CONFIGURATION (Cell 20 globals)
# ============================================================================

JSON_PATH = "benchmark_results_20260208_214002.json"

# Plot style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE_LARGE = (16, 12)
FIGSIZE_XLARGE = (20, 16)
DPI = 300


def plot_overview(conv_df, speed_df, overfit_df, output_dir):
    """Create 4-panel overview of training dynamics"""

    print("\n📊 Creating overview visualization...")

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    fig.suptitle('Training Dynamics Analysis Across Models and Datasets',
                 fontsize=16, fontweight='bold')

    # 1. Convergence epochs
    ax = axes[0, 0]
    pivot_conv = conv_df.pivot_table(values='convergence_epoch',
                                      index='model',
                                      columns='dataset',
                                      aggfunc='mean')
    pivot_conv.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Epochs to Convergence by Model', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('Epochs to Convergence', fontsize=10)
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Learning speed
    ax = axes[0, 1]
    pivot_speed = speed_df.pivot_table(values='learning_speed',
                                         index='model',
                                         columns='dataset',
                                         aggfunc='mean')
    pivot_speed.plot(kind='bar', ax=ax, width=0.8, colormap='viridis')
    ax.set_title('Learning Speed (Val Loss Reduction/Epoch)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('Avg Loss Reduction (First 5 Epochs)', fontsize=10)
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Overfitting score
    ax = axes[1, 0]
    pivot_overfit = overfit_df.pivot_table(values='overfitting_score',
                                            index='model',
                                            columns='dataset',
                                            aggfunc='mean')
    pivot_overfit.plot(kind='bar', ax=ax, width=0.8, colormap='coolwarm')
    ax.set_title('Overfitting Score (Train-Val Gap)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel('Overfitting Score', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Efficiency scatter
    ax = axes[1, 1]
    for dataset in conv_df['dataset'].unique():
        df_subset = conv_df[conv_df['dataset'] == dataset]
        ax.scatter(df_subset['convergence_epoch'],
                  df_subset['final_val_auc'],
                  s=df_subset['params'] / 1e5,
                  alpha=0.6,
                  label=dataset)

    ax.set_title('Training Efficiency: Convergence vs Performance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epochs to Convergence', fontsize=10)
    ax.set_ylabel('Final Validation AUC', fontsize=10)
    ax.legend(title='Dataset', fontsize=8)
    ax.grid(alpha=0.3)

    # Annotate interesting points
    for idx, row in conv_df.iterrows():
        if row['final_val_auc'] > 0.998 and row['convergence_epoch'] < 8:
            ax.annotate(row['model'],
                       xy=(row['convergence_epoch'], row['final_val_auc']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.7)

    plt.tight_layout()

    output_path = Path(output_dir) / 'training_dynamics_overview.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_learning_curves(results, output_dir, dataset_name=None):
    """Create detailed learning curves for a specific dataset"""

    # If no dataset specified, use the first one
    if dataset_name is None:
        dataset_name = list(results.keys())[0]

    print(f"\n📈 Creating learning curves for {dataset_name}...")

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    fig.suptitle(f'Learning Curves: {dataset_name}', fontsize=16, fontweight='bold')

    models_data = results[dataset_name]

    # 1. Validation Loss
    ax = axes[0, 0]
    for model_name, model_data in models_data.items():
        if 'history' not in model_data:
            continue
        history = model_data['history']
        epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(epochs, history['val_loss'], marker='o', markersize=4,
                label=model_name, linewidth=2, alpha=0.7)

    ax.set_title('Validation Loss Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Validation Loss', fontsize=10)
    ax.set_yscale('log')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # 2. Validation Accuracy
    ax = axes[0, 1]
    for model_name, model_data in models_data.items():
        if 'history' not in model_data:
            continue
        history = model_data['history']
        epochs = range(1, len(history['val_acc']) + 1)
        ax.plot(epochs, history['val_acc'], marker='o', markersize=4,
                label=model_name, linewidth=2, alpha=0.7)

    ax.set_title('Validation Accuracy Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Validation Accuracy', fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Train-Val Gap
    ax = axes[1, 0]
    for model_name, model_data in models_data.items():
        if 'history' not in model_data:
            continue
        history = model_data['history']
        min_len = min(len(history['train_acc']), len(history['val_acc']))
        epochs = range(1, min_len + 1)
        gap = [history['train_acc'][i] - history['val_acc'][i] for i in range(min_len)]
        ax.plot(epochs, gap, marker='o', markersize=4,
                label=model_name, linewidth=2, alpha=0.7)

    ax.set_title('Train-Validation Accuracy Gap (Overfitting)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy Gap (Train - Val)', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Validation AUC
    ax = axes[1, 1]
    for model_name, model_data in models_data.items():
        if 'history' not in model_data:
            continue
        history = model_data['history']
        if not history['val_auc']:
            continue
        epochs = range(1, len(history['val_auc']) + 1)
        ax.plot(epochs, history['val_auc'], marker='o', markersize=4,
                label=model_name, linewidth=2, alpha=0.7)

    ax.set_title('Validation AUC Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Validation AUC', fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / f'learning_curves_{dataset_name}.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_heatmaps(conv_df, output_dir):
    """Create heatmaps for convergence and performance"""

    print("\n🔥 Creating heatmaps...")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    # Heatmap 1: Convergence epochs
    ax = axes[0]
    pivot_conv = conv_df.pivot_table(values='convergence_epoch',
                                      index='model',
                                      columns='dataset',
                                      aggfunc='mean')
    sns.heatmap(pivot_conv, annot=True, fmt='.1f', cmap='YlOrRd_r',
                ax=ax, cbar_kws={'label': 'Epochs'}, linewidths=0.5)
    ax.set_title('Convergence Speed (Lower = Faster)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Model', fontsize=10)

    # Heatmap 2: Final performance
    ax = axes[1]
    pivot_perf = conv_df.pivot_table(values='final_val_auc',
                                      index='model',
                                      columns='dataset',
                                      aggfunc='mean')
    sns.heatmap(pivot_perf, annot=True, fmt='.4f', cmap='RdYlGn',
                ax=ax, cbar_kws={'label': 'AUC'}, linewidths=0.5,
                vmin=0.95, vmax=1.0)
    ax.set_title('Final Validation AUC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Model', fontsize=10)

    plt.tight_layout()

    output_path = Path(output_dir) / 'convergence_performance_heatmap.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_epoch_by_epoch(results, output_dir):
    """Create detailed epoch-by-epoch analysis for all datasets"""

    print("\n🔬 Creating epoch-by-epoch analysis...")

    dataset_names = list(results.keys())
    n_datasets = len(dataset_names)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig = plt.figure(figsize=FIGSIZE_XLARGE)
    gs = fig.add_gridspec(n_datasets, 4, hspace=0.3, wspace=0.3)

    for row, dataset_name in enumerate(dataset_names):
        models_data = results[dataset_name]

        # 1. Val Loss improvement per epoch
        ax1 = fig.add_subplot(gs[row, 0])
        for i, (model_name, model_data) in enumerate(models_data.items()):
            if 'history' not in model_data:
                continue
            history = model_data['history']
            val_loss = history['val_loss']

            if len(val_loss) > 1:
                improvements = [val_loss[j-1] - val_loss[j] for j in range(1, len(val_loss))]
                epochs = range(2, len(val_loss) + 1)
                ax1.plot(epochs, improvements, marker='o', markersize=3,
                        label=model_name, alpha=0.7, color=colors[i % len(colors)])

        ax1.set_title(f'{dataset_name}\nVal Loss Improvement per Epoch',
                     fontsize=10, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=8)
        ax1.set_ylabel('Loss Reduction', fontsize=8)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.grid(alpha=0.3)
        if row == 0:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

        # 2. Cumulative accuracy improvement
        ax2 = fig.add_subplot(gs[row, 1])
        for i, (model_name, model_data) in enumerate(models_data.items()):
            if 'history' not in model_data:
                continue
            history = model_data['history']
            val_acc = history['val_acc']

            if val_acc:
                cumulative_improvement = [acc - val_acc[0] for acc in val_acc]
                epochs = range(1, len(val_acc) + 1)
                ax2.plot(epochs, cumulative_improvement, marker='o', markersize=3,
                        label=model_name, alpha=0.7, color=colors[i % len(colors)])

        ax2.set_title(f'Cumulative Accuracy Gain\nfrom Epoch 1',
                     fontsize=10, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=8)
        ax2.set_ylabel('Accuracy Gain', fontsize=8)
        ax2.grid(alpha=0.3)

        # 3. Training stability (loss volatility)
        ax3 = fig.add_subplot(gs[row, 2])
        for i, (model_name, model_data) in enumerate(models_data.items()):
            if 'history' not in model_data:
                continue
            history = model_data['history']
            val_loss = history['val_loss']

            if len(val_loss) >= 3:
                changes = [abs(val_loss[j] - val_loss[j-1]) for j in range(1, len(val_loss))]
                rolling_volatility = []
                window = 3

                for j in range(len(changes)):
                    start = max(0, j - window + 1)
                    rolling_volatility.append(np.std(changes[start:j+1]))

                epochs = range(2, len(val_loss) + 1)
                ax3.plot(epochs, rolling_volatility, marker='o', markersize=3,
                        label=model_name, alpha=0.7, color=colors[i % len(colors)])

        ax3.set_title(f'Training Stability\n(Lower = More Stable)',
                     fontsize=10, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=8)
        ax3.set_ylabel('Loss Volatility', fontsize=8)
        ax3.set_yscale('log')
        ax3.grid(alpha=0.3)

        # 4. AUC trajectory
        ax4 = fig.add_subplot(gs[row, 3])
        for i, (model_name, model_data) in enumerate(models_data.items()):
            if 'history' not in model_data:
                continue
            history = model_data['history']
            val_auc = history.get('val_auc', [])

            if val_auc:
                epochs = range(1, len(val_auc) + 1)
                ax4.plot(epochs, val_auc, marker='o', markersize=3,
                        label=model_name, alpha=0.7, linewidth=2,
                        color=colors[i % len(colors)])

        ax4.set_title(f'AUC Trajectory', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=8)
        ax4.set_ylabel('Validation AUC', fontsize=8)
        ax4.grid(alpha=0.3)

    plt.suptitle('Epoch-by-Epoch Training Dynamics Across All Datasets',
                 fontsize=14, fontweight='bold', y=0.995)

    output_path = Path(output_dir) / 'epoch_by_epoch_analysis.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_convergence_patterns(results, conv_df, output_dir):
    """Advanced convergence pattern analysis"""

    print("\n🎯 Creating convergence patterns analysis...")

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    fig.suptitle('Convergence Pattern Analysis', fontsize=16, fontweight='bold')

    # 1. Epochs to reach 95% of final performance
    ax = axes[0, 0]
    reach_95_data = []

    for dataset_name, models_data in results.items():
        for model_name, model_data in models_data.items():
            if 'history' not in model_data:
                continue

            history = model_data['history']
            val_auc = history.get('val_auc', [])

            if not val_auc:
                continue

            final_auc = max(val_auc)
            target_auc = final_auc * 0.95

            epoch_95 = None
            for i, auc in enumerate(val_auc):
                if auc >= target_auc:
                    epoch_95 = i + 1
                    break

            if epoch_95:
                reach_95_data.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'epoch': epoch_95,
                    'final_auc': final_auc
                })

    reach_95_df = pd.DataFrame(reach_95_data)

    for dataset in reach_95_df['dataset'].unique():
        df_subset = reach_95_df[reach_95_df['dataset'] == dataset]
        df_avg = df_subset.groupby('model')['epoch'].mean().sort_values()
        df_avg.plot(kind='barh', ax=ax, alpha=0.7, label=dataset)

    ax.set_title('Epochs to Reach 95% of Final Performance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=10)
    ax.set_ylabel('Model', fontsize=10)
    ax.legend(title='Dataset', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

    # 2. Diminishing returns
    ax = axes[0, 1]
    dataset_names = list(results.keys())

    for dataset_name in dataset_names:
        models_data = results[dataset_name]
        model_name = 'resnet50'  # Use ResNet50 as example

        if model_name in models_data and 'history' in models_data[model_name]:
            history = models_data[model_name]['history']
            val_auc = history.get('val_auc', [])

            if len(val_auc) > 1:
                gains = [val_auc[i] - val_auc[i-1] for i in range(1, len(val_auc))]
                epochs = range(2, len(val_auc) + 1)
                ax.plot(epochs, gains, marker='o', markersize=5,
                       label=dataset_name, linewidth=2, alpha=0.7)

    ax.set_title('Diminishing Returns (ResNet50)\nAUC Gain per Additional Epoch',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('AUC Improvement', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(title='Dataset', fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Early vs late improvement
    ax = axes[1, 0]
    early_late_data = []

    for dataset_name, models_data in results.items():
        for model_name, model_data in models_data.items():
            if 'history' not in model_data:
                continue

            history = model_data['history']
            val_loss = history['val_loss']

            if len(val_loss) >= 6:
                early_improvement = val_loss[0] - val_loss[2]
                late_improvement = val_loss[-4] - val_loss[-1]

                early_late_data.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'early': early_improvement,
                    'late': late_improvement
                })

    early_late_df = pd.DataFrame(early_late_data)

    if len(early_late_df) > 0:
        ax.scatter(early_late_df['early'], early_late_df['late'],
                  s=100, alpha=0.6, c=range(len(early_late_df)), cmap='viridis')

        max_val = max(early_late_df['early'].max(), early_late_df['late'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal improvement')

        for idx, row in early_late_df.iterrows():
            if row['early'] > 0.05 or row['late'] > 0.01:
                ax.annotate(f"{row['model'][:6]}",
                           xy=(row['early'], row['late']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=6, alpha=0.7)

    ax.set_title('Early vs Late Training Improvement', fontsize=12, fontweight='bold')
    ax.set_xlabel('Early Improvement (Epochs 1-3)', fontsize=10)
    ax.set_ylabel('Late Improvement (Last 3 Epochs)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Training efficiency
    ax = axes[1, 1]
    efficiency_data = []

    for dataset_name, models_data in results.items():
        for model_name, model_data in models_data.items():
            if 'history' not in model_data:
                continue

            history = model_data['history']
            val_auc = history.get('val_auc', [])

            if val_auc:
                final_auc = max(val_auc)
                epochs_used = len(val_auc)
                efficiency = final_auc / epochs_used

                efficiency_data.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'efficiency': efficiency,
                    'final_auc': final_auc,
                    'epochs': epochs_used
                })

    efficiency_df = pd.DataFrame(efficiency_data)

    for dataset in efficiency_df['dataset'].unique():
        df_subset = efficiency_df[efficiency_df['dataset'] == dataset]
        df_sorted = df_subset.sort_values('efficiency', ascending=False).head(5)

        x = range(len(df_sorted))
        dataset_idx = dataset_names.index(dataset)
        ax.bar([i + dataset_idx * 0.15 for i in x],
               df_sorted['efficiency'],
               width=0.15,
               alpha=0.7,
               label=dataset)

    ax.set_title('Training Efficiency\n(Performance / Epochs)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Top 5 Models per Dataset', fontsize=10)
    ax.set_ylabel('Efficiency Score', fontsize=10)
    ax.legend(title='Dataset', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / 'convergence_patterns.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_summary_statistics(results, conv_df, speed_df, overfit_df):
    """Print comprehensive summary statistics"""

    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("="*70)

    # Fastest converging models
    print("\n🏃 FASTEST CONVERGING MODELS (by dataset):")
    for dataset in conv_df['dataset'].unique():
        df_subset = conv_df[conv_df['dataset'] == dataset]
        fastest = df_subset.nsmallest(3, 'convergence_epoch')
        print(f"\n{dataset}:")
        for idx, row in fastest.iterrows():
            print(f"  {row['model']:20s} - {row['convergence_epoch']:.0f} epochs "
                  f"(AUC: {row['final_val_auc']:.4f})")

    # Best first-epoch performers
    print("\n🚀 BEST FIRST-EPOCH PERFORMERS:")
    for dataset in speed_df['dataset'].unique():
        df_subset = speed_df[speed_df['dataset'] == dataset]
        best = df_subset.nlargest(3, 'epoch_1_val_acc')
        print(f"\n{dataset}:")
        for idx, row in best.iterrows():
            print(f"  {row['model']:20s} - Epoch 1 Acc: {row['epoch_1_val_acc']:.4f}")

    # Least overfitting
    print("\n✨ LEAST OVERFITTING MODELS:")
    for dataset in overfit_df['dataset'].unique():
        df_subset = overfit_df[overfit_df['dataset'] == dataset]
        best = df_subset.nsmallest(3, 'overfitting_score')
        print(f"\n{dataset}:")
        for idx, row in best.iterrows():
            print(f"  {row['model']:20s} - Overfit Score: {row['overfitting_score']:.6f}")

    # Efficiency champions
    print("\n🏆 EFFICIENCY CHAMPIONS (Fast Convergence + High Performance):")
    for dataset in conv_df['dataset'].unique():
        df_subset = conv_df[conv_df['dataset'] == dataset]
        df_subset = df_subset.copy()
        df_subset['conv_norm'] = 1 - (df_subset['convergence_epoch'] /
                                       df_subset['convergence_epoch'].max())
        df_subset['perf_norm'] = df_subset['final_val_auc']
        df_subset['efficiency'] = df_subset['conv_norm'] * 0.3 + df_subset['perf_norm'] * 0.7

        champions = df_subset.nlargest(3, 'efficiency')
        print(f"\n{dataset}:")
        for idx, row in champions.iterrows():
            print(f"  {row['model']:20s} - {row['convergence_epoch']:.0f} epochs, "
                  f"AUC: {row['final_val_auc']:.4f}")

    print("\n" + "="*70)


def save_data(conv_df, speed_df, overfit_df, output_dir):
    """Save processed data to CSV files"""

    print("\n💾 Saving data files...")

    # Convergence summary
    conv_summary = conv_df.pivot_table(
        values=['convergence_epoch', 'final_val_auc', 'best_val_loss'],
        index='model',
        columns='dataset',
        aggfunc='mean'
    )

    output_path = Path(output_dir) / 'convergence_summary.csv'
    conv_summary.to_csv(output_path)
    print(f"✓ Saved: {output_path}")

    # Full merged data
    all_data = conv_df.merge(speed_df, on=['dataset', 'model'])
    all_data = all_data.merge(overfit_df, on=['dataset', 'model'])

    output_path = Path(output_dir) / 'training_dynamics_full.csv'
    all_data.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")


if __name__ == "__main__":
    # Initialize
    config = Config()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Load progress tracker
    progress_tracker = ProgressTracker(config.PROGRESS_FILE)
    progress_tracker.print_status()

    # Get existing results or start fresh
    all_results = progress_tracker.get_results()

    # Main training loop
    for dataset_name in config.DATASETS:
        dataset_path = os.path.join(config.SPLIT_DIR, dataset_name)

        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            continue

        print(f"\n\n{'#'*70}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*70}")

        loaders, class_to_idx = get_data_loaders(
            dataset_path,
            batch_size=config.BATCH_SIZE,
            num_workers=2
        )

        num_classes = len(class_to_idx)
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {list(class_to_idx.keys())}")

        if dataset_name not in all_results:
            all_results[dataset_name] = {}

        for model_name in config.MODELS:
            # Skip if already completed
            if progress_tracker.is_completed(dataset_name, model_name):
                print(f"\n✓ Skipping {model_name} (already completed)")
                continue

            try:
                print(f"\n--- Training {model_name} ---")

                # Mark as started
                progress_tracker.mark_started(dataset_name, model_name)

                # Check if we're resuming
                resume_epoch = progress_tracker.get_resume_info(dataset_name, model_name)

                model = create_model(model_name, num_classes, pretrained=True)
                model = model.to(config.DEVICE)

                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Parameters: {total_params/1e6:.2f}M (trainable: {trainable_params/1e6:.2f}M)")

                criterion = nn.CrossEntropyLoss()

                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config.LEARNING_RATE,
                    weight_decay=config.WEIGHT_DECAY
                )

                scheduler = ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5
                )

                history, checkpoint_path = train_model(
                    model, loaders, criterion, optimizer, scheduler, config.DEVICE,
                    config.NUM_EPOCHS, model_name, dataset_name, config.CHECKPOINT_DIR,
                    progress_tracker, start_epoch=resume_epoch
                )

                # Load best model for evaluation
                final_model_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f"{model_name}_{dataset_name}_final.pt"
                )
                model.load_state_dict(torch.load(final_model_path, map_location=config.DEVICE))

                test_metrics, _, _ = evaluate_model(model, loaders['test'], config.DEVICE)

                print(f"\nTest Results:")
                for metric, value in test_metrics.items():
                    print(f"  {metric}: {value:.4f}")

                results = {
                    'test_metrics': test_metrics,
                    'history': history,
                    'params': trainable_params,
                }

                all_results[dataset_name][model_name] = results

                # Mark as completed
                progress_tracker.mark_completed(dataset_name, model_name, results)
                print(f"✓ {model_name} completed and saved")

                # Save intermediate results
                intermediate_path = os.path.join(
                    config.RESULTS_DIR,
                    f"results_intermediate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                save_results(all_results, intermediate_path)

            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                all_results[dataset_name][model_name] = {'error': str(e)}
                # Don't mark as completed so it can be retried

    results_path = os.path.join(
        config.RESULTS_DIR,
        f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_results(all_results, results_path)

    print(f"\n\n{'#'*70}")
    print(f"# SUMMARY")
    print(f"{'#'*70}")

    summary_df = []
    for dataset_name, models in all_results.items():
        for model_name, results in models.items():
            if 'test_metrics' in results:
                row = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'accuracy': results['test_metrics']['accuracy'],
                    'f1': results['test_metrics']['f1'],
                    'auc': results['test_metrics']['auc'],
                }
                summary_df.append(row)

    summary_df = pd.DataFrame(summary_df)
    summary_df = summary_df.sort_values('accuracy', ascending=False)

    print("\nTop Results (by Accuracy):")
    print(summary_df.head(10).to_string(index=False))

    summary_path = os.path.join(
        config.RESULTS_DIR,
        f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    print(f"\nResults saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")

    # Print final progress status
    progress_tracker.print_status()

    # Results table generation
    RESULTS_DIR = config.RESULTS_DIR

    # Find most recent results file
    results_file = max(Path(RESULTS_DIR).glob("benchmark_results_*.json"), key=os.path.getctime)
    with open(results_file, 'r') as f:
        all_results = json.load(f)

    data = []
    for dataset, models in all_results.items():
        for model, results in models.items():
            if 'test_metrics' in results:
                data.append({
                    'dataset': dataset, 'model': model,
                    'accuracy': results['test_metrics']['accuracy']*100,
                    'auc': results['test_metrics']['auc']*100
                })

    df = pd.DataFrame(data)

    models = ['resnet50','resnet101','vgg16','densenet121','densenet169','mobilenet_v2','efficientnet_b0','efficientnet_b4']
    datasets = ['MRI_tumor_binary_norm','MRI_tumor_multiclass_norm','MRI_ms_norm', 'CT_stroke_binary_norm']

    table = "| Model | " + " | ".join([d.replace('_norm','') for d in datasets]) + " |\n"
    table += "|-------|" + "---|"*len(datasets) + "\n"

    for model in models:
        row = f"| {model.replace('efficientnet','EffNet')} |"
        for dataset in datasets:
            acc = df[(df.model==model) & (df.dataset==dataset)].accuracy
            row += f" {acc.values[0]:.1f}% |" if len(acc)>0 else " - |"
        table += row + "\n"

    print("## FINAL TABLE")
    print(table)

    with open(os.path.join(RESULTS_DIR, "FINAL_TABLE.md"), "w") as f:
        f.write(table)
    print("✓ Saved: FINAL_TABLE.md")

    # Publication figures (Cell 19)
    results_file_19 = Path(RESULTS_DIR) / "benchmark_results_20260208_214002.json"

    with open(results_file_19, 'r') as f:
        all_results_19 = json.load(f)

    data19 = []
    for dataset, models in all_results_19.items():
        for model, results in models.items():
            if 'test_metrics' in results:
                data19.append({
                    'dataset': dataset.replace('_norm', ''),
                    'model': model.replace('efficientnet_', 'EffNet_'),
                    'accuracy': results['test_metrics']['accuracy'] * 100,
                    'f1': results['test_metrics']['f1'] * 100,
                    'auc': results['test_metrics']['auc'] * 100
                })

    df19 = pd.DataFrame(data19)
    print("Loaded your results:")
    print(df19.pivot(index='model', columns='dataset', values='accuracy').round(1))

    # FIG 1: HEATMAP (main paper figure)
    plt.figure(figsize=(9, 6))
    pivot = df19.pivot(index='model', columns='dataset', values='accuracy').round(1)
    sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.1f',
                cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.8},
                linewidths=0.5)
    plt.title('Neuroimaging Classification Benchmark (32 experiments)', fontsize=14, pad=20)
    plt.ylabel('Model', fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'heatmap_20.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # FIG 2: MODEL RANKING (average across tasks)
    model_avg = df19.groupby('model')['accuracy'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    bars = plt.barh(range(len(model_avg)), model_avg.values, color='steelblue')
    plt.yticks(range(len(model_avg)), model_avg.index)
    plt.xlabel('Average Accuracy (%)', fontsize=12)
    plt.title('Model Performance Ranking', fontsize=14)
    plt.axvline(90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
    plt.legend()
    # Add values on bars
    for i, v in enumerate(model_avg.values):
        plt.text(v + 0.2, i, f'{v:.1f}%', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_ranking_20.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n🏆 TOP MODELS:")
    print(model_avg.round(1))

    # FIG 3: TASK DIFFICULTY (boxplots)
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df19, x='dataset', y='accuracy', palette='Set2')
    plt.title('Task Difficulty Distribution', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.axhline(90, color='orange', linestyle=':', label='90%')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'task_difficulty_20.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # FIG 4: PARAMETER EFFICIENCY
    model_params = {
        'resnet50': 25.6, 'resnet101': 44.5, 'vgg16': 138.4,
        'densenet121': 8.0, 'densenet169': 14.1,
        'mobilenet_v2': 3.5, 'EffNet_b0': 5.3, 'EffNet_b4': 19.3
    }
    df19['params_M'] = df19['model'].map(model_params)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df19, x='params_M', y='accuracy', hue='model', s=100)
    plt.title('Accuracy vs Parameter Count')
    plt.xlabel('Parameters (millions)')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'param_efficiency_20.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print("\n📊 SUMMARY STATS FROM YOUR JSON:")
    print(df19.groupby('dataset')['accuracy'].agg(['mean', 'std']).round(1))

    # Training dynamics analysis (Cell 21)
    print("="*70)
    print("TRAINING DYNAMICS ANALYSIS")
    print("="*70)

    OUTPUT_DIR = os.path.join(config.BASE_DIR, "training_analysis_outputs")
    json_path_analysis = os.path.join(RESULTS_DIR, "benchmark_results_20260208_214002.json")

    output_dir = OUTPUT_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load and process data
    results, conv_df, speed_df, overfit_df = load_and_process_data(json_path_analysis)

    # Generate all visualizations
    plot_overview(conv_df, speed_df, overfit_df, output_dir)

    # Create learning curves for each dataset
    for dataset_name in results.keys():
        plot_learning_curves(results, output_dir, dataset_name)

    plot_heatmaps(conv_df, output_dir)
    plot_epoch_by_epoch(results, output_dir)
    plot_convergence_patterns(results, conv_df, output_dir)

    # Print statistics
    print_summary_statistics(results, conv_df, speed_df, overfit_df)

    # Save data files
    save_data(conv_df, speed_df, overfit_df, output_dir)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {output_dir}/")
    print("="*70)
