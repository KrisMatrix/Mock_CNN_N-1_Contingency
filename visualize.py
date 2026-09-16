"""
visualize.py
Visualization suite for N-1 Contingency Recommendation CNN.
Generates heatmaps of 3 input channels, ground truth vs predicted matrices,
training history curves, and ROC/PR curves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from model import load_dataset
from evaluate import evaluate_cnn_model


def plot_input_channels(x_sample, save_path='plots/01_input_channels_heatmap.png'):
    """
    Plots heatmaps of 3 input matrix channels for sample 0:
    Channel 1: Admittance magnitude |Y_bus|
    Channel 2: Real Power Matrix P
    Channel 3: Reactive Power Matrix Q
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = [
        "Channel 1: Admittance Matrix |Y_bus|",
        "Channel 2: Real Power Matrix P",
        "Channel 3: Reactive Power Matrix Q"
    ]
    cmaps = ['viridis', 'plasma', 'coolwarm']

    for i in range(3):
        im = axes[i].imshow(x_sample[:, :, i], cmap=cmaps[i], aspect='equal')
        axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Bus Index (0-34)")
        axes[i].set_ylabel("Bus Index (0-34)")
        fig.colorbar(im, ax=axes[i], shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Input Channels Heatmap plot to: {save_path}")


def plot_contingency_predictions(y_true, prob_matrix, save_path='plots/02_ground_truth_vs_predicted_heatmap.png'):
    """
    Plots side-by-side comparison of Ground Truth N-1 Matrix vs Predicted Probability Matrix vs Binary Prediction Matrix.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    binary_pred = (prob_matrix >= 0.5).astype(np.float32)

    im0 = axes[0].imshow(y_true, cmap='Blues', aspect='equal', vmin=0, vmax=1)
    axes[0].set_title("Ground Truth N-1 Matrix Y", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Bus Index")
    axes[0].set_ylabel("Bus Index")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(prob_matrix, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
    axes[1].set_title("Predicted Recommendation Probabilities", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Bus Index")
    axes[1].set_ylabel("Bus Index")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(binary_pred, cmap='Greens', aspect='equal', vmin=0, vmax=1)
    axes[2].set_title("Binary Recommendation Threshold (p >= 0.5)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Bus Index")
    axes[2].set_ylabel("Bus Index")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Contingency Prediction Comparison plot to: {save_path}")


def plot_training_curves(history_path='data/training_history.npz', save_path='plots/03_training_curves.png'):
    """
    Plots training and validation loss, accuracy, F1-score, and AUC over epochs.
    """
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return

    hist = np.load(history_path)
    epochs = np.arange(1, len(hist['loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, hist['loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, hist['val_loss'], 'r--s', label='Val Loss', linewidth=2)
    axes[0, 0].set_title("Weighted BCE Loss", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(epochs, hist['accuracy'] * 100, 'b-o', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, hist['val_accuracy'] * 100, 'r--s', label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title("Binary Accuracy (%)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend()

    # F1-Score
    axes[1, 0].plot(epochs, hist['f1_score'], 'b-o', label='Train F1', linewidth=2)
    axes[1, 0].plot(epochs, hist['val_f1_score'], 'r--s', label='Val F1', linewidth=2)
    axes[1, 0].set_title("F1-Score", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].legend()

    # ROC-AUC
    axes[1, 1].plot(epochs, hist['auc'], 'b-o', label='Train AUC', linewidth=2)
    axes[1, 1].plot(epochs, hist['val_auc'], 'r--s', label='Val AUC', linewidth=2)
    axes[1, 1].set_title("ROC-AUC Score", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC")
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].legend()

    plt.suptitle("ContingencyCNN Training & Validation History (25 Epochs)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Training Curves plot to: {save_path}")


def plot_roc_pr_curves(y_true_flat, y_prob_flat, save_path='plots/04_roc_pr_curves.png'):
    """
    Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
    roc_auc_val = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc="lower right")

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true_flat, y_prob_flat)
    pr_auc_val = auc(rec, prec)
    axes[1].plot(rec, prec, color='green', lw=2, label=f'PR Curve (AUC = {pr_auc_val:.4f})')
    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC & Precision-Recall Curves plot to: {save_path}")


def visualize_results(model_path='checkpoints/best_model.keras', dataset_path='data/dataset.npz'):
    """
    Runs evaluation and generates all 4 visualization plots.
    """
    # 1. Load Data sample 0 for channel visualization
    (_, _), (_, _), (x_test, y_test) = load_dataset(dataset_path)
    plot_input_channels(x_test[0])

    # 2. Evaluate model and get predictions
    results, probs, y_test_mats = evaluate_cnn_model(model_path, dataset_path)
    plot_contingency_predictions(y_test_mats[0], probs[0])

    # 3. Training curves
    plot_training_curves()

    # 4. ROC and PR curves
    plot_roc_pr_curves(y_test_mats.flatten(), probs.flatten())

    print("\nAll Phase 5 Visualizations generated successfully in 'plots/' directory!")


if __name__ == '__main__':
    visualize_results()
