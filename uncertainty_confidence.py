"""
Comprehensive Uncertainty, Confidence, and Explainability Experiments
for CLIP and BiomedCLIP Models

This module implements state-of-the-art methods for:
1. Uncertainty Quantification (Conformal Prediction, Entropy)
2. Confidence Calibration (Temperature Scaling, Platt Scaling, Isotonic Regression)
3. Explainability (Attention Visualization, GradCAM, Token Attribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import open_clip
from PIL import Image
import cv2
from pathlib import Path


# ============================================================================
# PART 1: UNCERTAINTY QUANTIFICATION METHODS
# ============================================================================

class ConformalPrediction:
    """
    Conformal Prediction for Vision-Language Models
    
    Provides prediction sets with guaranteed coverage using adaptive prediction sets (APS).
    Reference: "Uncertainty-Aware Evaluation for Vision-Language Models" (2024)
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Desired error rate (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.qhat = None
        
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Calibrate on validation set to compute quantile threshold.
        
        Args:
            logits: Model output logits [N, num_classes]
            labels: Ground truth labels [N]
        """
        # Compute softmax scores
        probs = F.softmax(logits, dim=-1)
        
        # Compute non-conformity scores using APS (cumulative sorted probabilities)
        n = len(labels)
        cal_scores = []
        
        for i in range(n):
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
            
            # Find position of true label in sorted list
            true_label_pos = (sorted_indices == labels[i]).nonzero(as_tuple=True)[0].item()
            
            # Cumulative sum up to and including true label
            score = sorted_probs[:true_label_pos + 1].sum().item()
            cal_scores.append(score)
        
        # Compute corrected quantile
        cal_scores = np.array(cal_scores)
        n_cal = len(cal_scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        self.qhat = np.quantile(cal_scores, q_level, method='higher')
        
        return self.qhat
    
    def predict(self, logits: torch.Tensor) -> List[List[int]]:
        """
        Generate prediction sets for test samples.
        
        Args:
            logits: Model output logits [N, num_classes]
            
        Returns:
            List of prediction sets (list of class indices for each sample)
        """
        if self.qhat is None:
            raise ValueError("Must calibrate before predicting")
        
        probs = F.softmax(logits, dim=-1)
        prediction_sets = []
        
        for prob in probs:
            sorted_probs, sorted_indices = torch.sort(prob, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            
            # Include classes until cumulative probability exceeds threshold
            set_size = (cumsum <= self.qhat).sum().item() + 1
            pred_set = sorted_indices[:set_size].tolist()
            prediction_sets.append(pred_set)
        
        return prediction_sets
    
    def evaluate_coverage_and_size(self, prediction_sets: List[List[int]], 
                                   labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate empirical coverage and average set size.
        """
        coverage = np.mean([labels[i].item() in pred_set 
                          for i, pred_set in enumerate(prediction_sets)])
        avg_size = np.mean([len(pred_set) for pred_set in prediction_sets])
        
        return {
            'coverage': coverage,
            'avg_set_size': avg_size,
            'target_coverage': 1 - self.alpha
        }


# ============================================================================
# PART 2: CONFIDENCE CALIBRATION METHODS
# ============================================================================

class TemperatureScaling:
    """
    Temperature scaling for calibrating neural network confidences.
    
    A simple post-hoc calibration method that learns a single temperature parameter.
    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """
    
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1))
        
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, 
                 lr: float = 0.01, max_iter: int = 100):
        """
        Learn optimal temperature on validation set.
        
        Args:
            logits: Uncalibrated logits [N, num_classes]
            labels: Ground truth labels [N]
            lr: Learning rate
            max_iter: Maximum optimization iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        return self.temperature.item()
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply learned temperature to calibrate predictions.
        """
        return F.softmax(logits / self.temperature, dim=-1)


class PlattScaling:
    """
    Platt scaling (logistic regression on output scores).
    
    Fits a logistic regression model on validation set to calibrate probabilities.
    """
    
    def __init__(self):
        self.model = LogisticRegression()
        
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Fit Platt scaling on validation set.
        """
        # For multi-class, use one-vs-rest
        scores = F.softmax(logits, dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        self.model.fit(scores, labels_np)
        
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Platt scaling to calibrate predictions.
        """
        scores = F.softmax(logits, dim=-1).cpu().numpy()
        calibrated = self.model.predict_proba(scores)
        return torch.from_numpy(calibrated)


class IsotonicRegressionCalibration:
    """
    Isotonic regression for calibration.
    
    Non-parametric calibration method that fits isotonic regression to validation scores.
    """
    
    def __init__(self):
        self.calibrators = []
        
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Fit isotonic regression for each class.
        """
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        num_classes = probs.shape[1]
        
        self.calibrators = []
        for c in range(num_classes):
            # Create binary labels for this class
            binary_labels = (labels_np == c).astype(int)
            
            # Fit isotonic regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(probs[:, c], binary_labels)
            self.calibrators.append(calibrator)
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply isotonic calibration.
        """
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        calibrated = np.zeros_like(probs)
        
        for c, calibrator in enumerate(self.calibrators):
            calibrated[:, c] = calibrator.predict(probs[:, c])
        
        # Normalize to sum to 1
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        return torch.from_numpy(calibrated)


class CalibrationMetrics:
    """
    Comprehensive calibration metrics for evaluating confidence estimation.
    """
    
    @staticmethod
    def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, 
                                  num_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error (ECE).
        """
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    @staticmethod
    def maximum_calibration_error(probs: torch.Tensor, labels: torch.Tensor,
                                 num_bins: int = 15) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        """
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)
        
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        mce = 0.0
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, torch.abs(avg_confidence_in_bin - accuracy_in_bin).item())
        
        return mce
    
    @staticmethod
    def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Brier score (mean squared error of probabilistic predictions).
        """
        # Convert labels to one-hot
        num_classes = probs.shape[1]
        labels_one_hot = F.one_hot(labels, num_classes).float()
        
        return torch.mean((probs - labels_one_hot) ** 2).item()
    
    @staticmethod
    def reliability_diagram(probs: torch.Tensor, labels: torch.Tensor, 
                          num_bins: int = 10, save_path: Optional[str] = None):
        """
        Generate and optionally save reliability diagram.
        """
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)
        
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracies.append(accuracies[in_bin].float().mean().item())
                bin_confidences.append(confidences[in_bin].mean().item())
                bin_counts.append(in_bin.sum().item())
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper).item() / 2)
                bin_counts.append(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Reliability bars
        ax.bar(bin_confidences, bin_accuracies, width=1/num_bins, 
               alpha=0.6, edgecolor='black', label='Model')
        
        # Gap visualization
        for conf, acc in zip(bin_confidences, bin_accuracies):
            ax.plot([conf, conf], [acc, conf], 'r-', alpha=0.3)
        
        ax.set_xlabel('Confidence', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_title('Reliability Diagram', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, (bin_confidences, bin_accuracies, bin_counts)


# ============================================================================
# PART 3: EXPLAINABILITY METHODS
# ============================================================================

class AttentionVisualizer:
    """
    Visualize attention patterns in vision-language models.
    
    Extracts and visualizes attention weights from transformer layers to understand
    what the model focuses on when making predictions.
    """
    
    @staticmethod
    def extract_attention_maps(model, images: torch.Tensor, 
                              layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention maps from specified transformer layer.
        
        Args:
            model: CLIP model
            images: Input images [B, C, H, W]
            layer_idx: Which transformer layer to visualize (-1 for last)
            
        Returns:
            Attention maps [B, num_heads, num_patches, num_patches]
        """
        attention_maps = []
        
        # Register hook to capture attention
        def hook_fn(module, input, output):
            # output is typically (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) > 1:
                attention_maps.append(output[1])
        
        # Register hook on attention layer
        if hasattr(model.visual, 'transformer'):
            target_layer = model.visual.transformer.resblocks[layer_idx].attn
        else:
            target_layer = list(model.visual.children())[layer_idx]
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = model.encode_image(images)
        
        handle.remove()
        
        return attention_maps[0] if attention_maps else None
    
    @staticmethod
    def visualize_attention(image: Image.Image, attention_map: torch.Tensor,
                           save_path: Optional[str] = None):
        """
        Visualize attention map overlaid on original image.
        """
        # Average attention across heads and compute mean attention to CLS token
        # Shape: [num_heads, num_patches, num_patches]
        if attention_map.dim() == 3:
            # Average over heads
            attention_map = attention_map.mean(0)
        
        # Get attention from CLS token (first token) to all patches
        cls_attention = attention_map[0, 1:]  # Exclude CLS to CLS
        
        # Reshape to spatial dimensions
        grid_size = int(np.sqrt(len(cls_attention)))
        attention_map_2d = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
        
        # Resize to image dimensions
        img_array = np.array(image)
        attention_resized = cv2.resize(attention_map_2d, 
                                      (img_array.shape[1], img_array.shape[0]))
        
        # Normalize
        attention_resized = (attention_resized - attention_resized.min()) / \
                           (attention_resized.max() - attention_resized.min())
        
        # Create heatmap
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(attention_resized, cmap='jet')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(img_array)
        axes[2].imshow(attention_resized, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class GradCAMForCLIP:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for CLIP models.
    
    Computes importance of image regions for specific text prompts using gradients.
    Reference: "CLIP Surgery" and "Explainability for CLIP"
    """
    
    def __init__(self, model, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: CLIP model
            target_layer: Layer to compute gradients from (default: last conv layer)
        """
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Automatically select target layer if not provided
        if target_layer is None:
            if hasattr(model.visual, 'transformer'):
                # OpenCLIP native ViT — last ResidualAttentionBlock
                self.target_layer = model.visual.transformer.resblocks[-1]
            elif hasattr(model.visual, 'trunk') and hasattr(model.visual.trunk, 'blocks'):
                # Timm-based ViT (e.g. BiomedCLIP) — last timm Block
                self.target_layer = model.visual.trunk.blocks[-1]
            else:
                # CNN fallback
                self.target_layer = list(model.visual.children())[-2]
        else:
            self.target_layer = target_layer
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        # Some blocks return tuples (output, extra); take the tensor
        self.activations = output[0] if isinstance(output, tuple) else output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, image: torch.Tensor, text_embedding: torch.Tensor) -> np.ndarray:
        """
        Generate Class Activation Map for given image and text.
        
        Args:
            image: Input image tensor [1, C, H, W]
            text_embedding: Text embedding for target concept [1, embed_dim]
            
        Returns:
            CAM heatmap [H, W]
        """
        self.model.eval()
        
        # Forward pass
        image_features = self.model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity
        similarity = (image_features @ text_embedding.T).squeeze()
        
        # Backward pass
        self.model.zero_grad()
        similarity.backward()
        
        # Compute weighted activation map
        # For ViT: [B, num_patches, dim]
        # For CNN: [B, C, H, W]
        if self.gradients.dim() == 3:  # ViT
            gradients_no_cls = self.gradients[:, 1:, :]  # [B, 49, dim]
            activations_no_cls = self.activations[:, 1:, :]  # [B, 49, dim]

            weights = gradients_no_cls.mean(dim=1)
            cam = (weights.unsqueeze(1) * activations_no_cls).sum(dim=-1)  # 49 tokens
            num_patches = cam.shape[1]  # 49
            grid_size = int(np.sqrt(num_patches))  # 7
            cam = cam.reshape(1, grid_size, grid_size)  # SUCCESS: 49 → 7×7 ✓
        else:  # CNN
            # Global average pooling of gradients
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            
            # Weighted combination
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize_cam(self, image: Image.Image, cam: np.ndarray,
                     save_path: Optional[str] = None):
        """
        Visualize CAM overlaid on original image.
        """
        # Resize CAM to image dimensions
        img_array = np.array(image)
        cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        axes[2].imshow(img_array)
        axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def __del__(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


class TokenAttributionAnalysis:
    """
    Analyze token-level contributions in vision-language models.
    
    Uses integrated gradients to attribute predictions to specific image patches
    and text tokens.
    """
    
    @staticmethod
    def integrated_gradients(model, image: torch.Tensor, text_embedding: torch.Tensor,
                           baseline: Optional[torch.Tensor] = None,
                           steps: int = 50) -> torch.Tensor:
        """
        Compute integrated gradients for input attribution.
        
        Args:
            model: CLIP model
            image: Input image [1, C, H, W]
            text_embedding: Text embedding [1, embed_dim]
            baseline: Baseline image (default: zeros)
            steps: Number of integration steps
            
        Returns:
            Attribution map [C, H, W]
        """
        if baseline is None:
            baseline = torch.zeros_like(image)
        
        # Generate interpolated images
        alphas = torch.linspace(0, 1, steps).to(image.device)
        interpolated_images = baseline + alphas.view(-1, 1, 1, 1) * (image - baseline)
        
        # Compute gradients for each interpolated image
        gradients = []
        
        for interp_image in interpolated_images:
            interp_image = interp_image.unsqueeze(0).requires_grad_(True)
            
            # Forward pass
            image_features = model.encode_image(interp_image)
            image_features = F.normalize(image_features, dim=-1)
            similarity = (image_features @ text_embedding.T).sum()
            
            # Backward pass
            model.zero_grad()
            similarity.backward()
            
            gradients.append(interp_image.grad.detach())
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Multiply by input difference
        attribution = (image - baseline) * avg_gradients
        
        return attribution.squeeze()
    
    @staticmethod
    def visualize_attribution(image: Image.Image, attribution: torch.Tensor,
                            save_path: Optional[str] = None):
        """
        Visualize token attribution map.
        """
        # Convert to numpy and take absolute values
        attribution_map = attribution.abs().sum(dim=0).cpu().numpy()
        
        # Normalize
        attribution_map = (attribution_map - attribution_map.min()) / \
                         (attribution_map.max() - attribution_map.min() + 1e-8)
        
        # Create visualization
        img_array = np.array(image)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(attribution_map, cmap='hot')
        axes[1].set_title('Attribution Map')
        axes[1].axis('off')
        
        axes[2].imshow(img_array)
        axes[2].imshow(attribution_map, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# ============================================================================
# COMPREHENSIVE EVALUATION PIPELINE
# ============================================================================

class ComprehensiveEvaluator:
    """
    Complete pipeline for uncertainty, confidence, and explainability evaluation.
    """
    
    def __init__(self, model, device='cuda', text_embeddings: Optional[torch.Tensor] = None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.text_embeddings = text_embeddings

        # Initialize all methods
        self.conformal = ConformalPrediction(alpha=0.1)
        self.temp_scaling = TemperatureScaling()
        self.platt_scaling = PlattScaling()
        self.isotonic = IsotonicRegressionCalibration()

        # Explainability tools
        self.grad_cam = GradCAMForCLIP(model)
        
    def set_text_embeddings(self, text_embeddings: torch.Tensor):
        """Set text embeddings for CLIP-based logit computation."""
        self.text_embeddings = text_embeddings

    def run_full_evaluation(self,
                          train_loader,
                          val_loader,
                          test_loader,
                          class_names: List[str],
                          output_dir: Path):
        """
        Run complete evaluation pipeline.
        
        Args:
            train_loader: Training data (for deep ensembles if needed)
            val_loader: Validation data (for calibration)
            test_loader: Test data (for evaluation)
            class_names: Names of classes
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        print("="*80)
        print("STEP 1: Collecting predictions on validation set")
        print("="*80)
        
        val_logits, val_labels = self._collect_predictions(val_loader)
        
        print("\nSTEP 2: Calibrating models")
        print("="*80)
        
        # Temperature scaling
        print("Calibrating temperature scaling...")
        temp = self.temp_scaling.calibrate(val_logits, val_labels)
        results['temperature'] = temp
        print(f"  Optimal temperature: {temp:.4f}")
        
        # Platt scaling
        print("Calibrating Platt scaling...")
        self.platt_scaling.calibrate(val_logits, val_labels)
        
        # Isotonic regression
        print("Calibrating isotonic regression...")
        self.isotonic.calibrate(val_logits, val_labels)
        
        # Conformal prediction
        print("Calibrating conformal prediction...")
        qhat = self.conformal.calibrate(val_logits, val_labels)
        results['conformal_threshold'] = qhat
        print(f"  Conformal threshold: {qhat:.4f}")
        
        print("\nSTEP 3: Evaluating on test set")
        print("="*80)
        
        test_logits, test_labels = self._collect_predictions(test_loader)
        
        # Baseline metrics
        baseline_probs = F.softmax(test_logits, dim=-1)
        results['baseline'] = self._compute_metrics(baseline_probs, test_labels, "Baseline")
        
        # Temperature scaled
        temp_probs = self.temp_scaling.apply(test_logits)
        results['temperature_scaled'] = self._compute_metrics(
            temp_probs, test_labels, "Temperature Scaled"
        )
        
        # Platt scaled
        platt_probs = self.platt_scaling.apply(test_logits)
        results['platt_scaled'] = self._compute_metrics(
            platt_probs, test_labels, "Platt Scaled"
        )
        
        # Isotonic
        isotonic_probs = self.isotonic.apply(test_logits)
        results['isotonic'] = self._compute_metrics(
            isotonic_probs, test_labels, "Isotonic"
        )
        
        # Conformal prediction
        prediction_sets = self.conformal.predict(test_logits)
        conf_metrics = self.conformal.evaluate_coverage_and_size(prediction_sets, test_labels)
        results['conformal'] = conf_metrics
        print(f"\nConformal Prediction:")
        print(f"  Coverage: {conf_metrics['coverage']:.4f} (target: {conf_metrics['target_coverage']:.4f})")
        print(f"  Avg set size: {conf_metrics['avg_set_size']:.2f}")
        
        print("\nSTEP 4: Generating reliability diagrams")
        print("="*80)
        
        # Generate reliability diagrams
        CalibrationMetrics.reliability_diagram(
            baseline_probs, test_labels,
            save_path=output_dir / 'reliability_baseline.png'
        )
        CalibrationMetrics.reliability_diagram(
            temp_probs, test_labels,
            save_path=output_dir / 'reliability_temperature.png'
        )
        
        print("\nSTEP 5: Explainability analysis (sample images)")
        print("="*80)
        
        # Analyze first few test samples
        self._generate_explainability_visualizations(
            test_loader, class_names, output_dir, num_samples=5
        )
        
        # Save results
        import json
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print(f"Results saved to: {output_dir}")
        print("="*80)
        
        return results
    
    def _collect_predictions(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect model logits and labels from dataloader.

        Requires text_embeddings to be set (via __init__ or set_text_embeddings)
        so that similarity logits [N, num_classes] are returned, not raw features.
        """
        if self.text_embeddings is None:
            raise ValueError(
                "text_embeddings must be set before collecting predictions. "
                "Use set_text_embeddings() or pass text_embeddings to __init__."
            )

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)

                # Get image features and normalize
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)

                # Compute similarity logits against text embeddings
                logits = image_features @ self.text_embeddings.T * 100.0

                all_logits.append(logits.cpu())
                all_labels.append(labels)

        return torch.cat(all_logits), torch.cat(all_labels)
    
    def _compute_metrics(self, probs: torch.Tensor, labels: torch.Tensor, 
                        name: str) -> Dict[str, float]:
        """Compute and print calibration metrics."""
        metrics = {
            'ece': CalibrationMetrics.expected_calibration_error(probs, labels),
            'mce': CalibrationMetrics.maximum_calibration_error(probs, labels),
            'brier': CalibrationMetrics.brier_score(probs, labels),
            'nll': log_loss(labels.cpu().numpy(), probs.cpu().numpy())
        }
        
        print(f"\n{name}:")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  MCE: {metrics['mce']:.4f}")
        print(f"  Brier: {metrics['brier']:.4f}")
        print(f"  NLL: {metrics['nll']:.4f}")
        
        return metrics
    
    def _generate_explainability_visualizations(self, dataloader, class_names: List[str],
                                               output_dir: Path, num_samples: int = 5):
        """Generate Grad-CAM explainability visualizations for sample images.

        Requires text_embeddings to be set for CLIP-based Grad-CAM computation.
        """
        if self.text_embeddings is None:
            print("  Skipping explainability: text_embeddings not set.")
            return

        explainability_dir = output_dir / 'explainability'
        explainability_dir.mkdir(parents=True, exist_ok=True)

        sample_count = 0

        for images, labels in dataloader:
            if sample_count >= num_samples:
                break

            for i in range(min(len(images), num_samples - sample_count)):
                image = images[i:i+1].to(self.device)
                label = labels[i].item()

                # Get prediction
                with torch.no_grad():
                    img_feat = self.model.encode_image(image)
                    img_feat = F.normalize(img_feat, dim=-1)
                    logits = img_feat @ self.text_embeddings.T * 100.0
                    pred_idx = logits.argmax(dim=-1).item()

                # Generate Grad-CAM for predicted class
                text_emb = self.text_embeddings[pred_idx:pred_idx + 1]
                cam = self.grad_cam.generate_cam(image, text_emb)

                # Create a simple PIL image from the tensor for visualization
                img_np = images[i].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                pil_image = Image.fromarray((img_np * 255).astype(np.uint8))

                fig = self.grad_cam.visualize_cam(pil_image, cam)

                true_class = class_names[label] if label < len(class_names) else str(label)
                pred_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                correct = "correct" if pred_idx == label else "incorrect"

                fig.suptitle(
                    f"True: {true_class} | Pred: {pred_class} ({correct})",
                    fontsize=12, y=1.02
                )

                save_path = explainability_dir / f'gradcam_sample_{sample_count}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

                sample_count += 1
                if sample_count >= num_samples:
                    break

        print(f"  Generated {sample_count} Grad-CAM visualizations in {explainability_dir}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def load_clip_model(model_name: str = 'ViT-B-32', pretrained: str = 'openai'):
    """Load CLIP or BiomedCLIP model."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    return model, preprocess


def load_biomedclip_model():
    """Load BiomedCLIP model from HuggingFace."""
    import json
    from urllib.request import urlopen
    from huggingface_hub import hf_hub_download
    
    # Download model files
    hf_hub_download(
        repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        filename="open_clip_pytorch_model.bin",
        local_dir="checkpoints"
    )
    hf_hub_download(
        repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        filename="open_clip_config.json",
        local_dir="checkpoints"
    )
    
    # Load model
    model_name = "biomedclip_local"
    with open("checkpoints/open_clip_config.json", "r") as f:
        config = json.load(f)
    
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]
    
    from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
    if model_name not in _MODEL_CONFIGS:
        _MODEL_CONFIGS[model_name] = model_cfg
    
    tokenizer = open_clip.get_tokenizer(model_name)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained="checkpoints/open_clip_pytorch_model.bin"
    )
    
    return model, tokenizer, preprocess


if __name__ == "__main__":
    print("Uncertainty, Confidence, and Explainability Toolkit for CLIP/BiomedCLIP")
    print("="*80)
    print("\nThis module provides implementations of:")
    print("  - Conformal Prediction")
    print("  - Temperature/Platt/Isotonic Calibration")
    print("  - Attention Visualization")
    print("  - Grad-CAM")
    print("  - Token Attribution")
    print("\nSee documentation in each class for usage examples.")
