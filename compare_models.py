"""
compare_models.py
Comparative analysis and visualization script comparing 2D CNN vs Graph Neural Network (GNN)
for N-1 contingency stress-testing recommendation across Decoupled DC Powerflow and Full AC Contingency Solvers.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def compare_single_mode(cnn_json, gnn_json, mode_name, save_path):
    """
    Evaluates and plots CNN vs GNN for a specific power flow solver mode ('DC' or 'AC').
    """
    if not os.path.exists(cnn_json) or not os.path.exists(gnn_json):
        print(f"Skipping {mode_name} comparison: evaluation JSON files not found.")
        return None, None

    with open(cnn_json, 'r') as f:
        cnn = json.load(f)
    with open(gnn_json, 'r') as f:
        gnn = json.load(f)

    print("\n" + "=" * 75)
    print(f"   MODEL PERFORMANCE COMPARISON [{mode_name.upper()} POWER FLOW]: 2D CNN vs GNN")
    print("=" * 75)
    print(f"{'Metric':<20} | {'2D CNN (Grid Input)':<22} | {'GNN (Graph Input)':<22} | {'Delta (GNN - CNN)':<18}")
    print("-" * 75)

    metrics_keys = [
        ('test_loss', 'Test Loss', False),
        ('accuracy', 'Accuracy (%)', True),
        ('precision', 'Precision (%)', True),
        ('recall', 'Recall (%)', True),
        ('f1_score', 'F1-Score', False),
        ('roc_auc', 'ROC-AUC Score', False)
    ]

    for key, name, is_percent in metrics_keys:
        cnn_val = cnn[key]
        gnn_val = gnn[key]

        if is_percent:
            cnn_str = f"{cnn_val * 100:.2f}%"
            gnn_str = f"{gnn_val * 100:.2f}%"
            delta = (gnn_val - cnn_val) * 100
            delta_str = f"{delta:+.2f}%"
        else:
            cnn_str = f"{cnn_val:.4f}"
            gnn_str = f"{gnn_val:.4f}"
            delta = gnn_val - cnn_val
            delta_str = f"{delta:+.4f}"

        print(f"{name:<20} | {cnn_str:<22} | {gnn_str:<22} | {delta_str:<18}")

    print("=" * 75)

    # Plot Comparison
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    metric_names = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score', 'ROC-AUC']
    cnn_scores = [cnn['accuracy'] * 100, cnn['precision'] * 100, cnn['recall'] * 100, cnn['f1_score'], cnn['roc_auc']]
    gnn_scores = [gnn['accuracy'] * 100, gnn['precision'] * 100, gnn['recall'] * 100, gnn['f1_score'], gnn['roc_auc']]

    x = np.arange(len(metric_names))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, cnn_scores, width, label='2D CNN', color='royalblue')
    rects2 = axes[0].bar(x + width/2, gnn_scores, width, label='Graph Neural Network (GNN)', color='darkorchid')

    axes[0].set_ylabel('Score / Percentage')
    axes[0].set_title(f'CNN vs GNN [{mode_name.upper()} Power Flow] Metrics', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_names, fontsize=10)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5, axis='y')

    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f'{h:.2f}' if h < 1.5 else f'{h:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f'{h:.2f}' if h < 1.5 else f'{h:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Confusion matrix error breakdown
    cnn_cm = cnn['confusion_matrix']
    gnn_cm = gnn['confusion_matrix']
    cm_labels = ['True Positives', 'False Positives', 'False Negatives']
    cnn_cm_vals = [cnn_cm['true_positives'], cnn_cm['false_positives'], cnn_cm['false_negatives']]
    gnn_cm_vals = [gnn_cm['true_positives'], gnn_cm['false_positives'], gnn_cm['false_negatives']]

    x_cm = np.arange(len(cm_labels))
    axes[1].bar(x_cm - width/2, cnn_cm_vals, width, label='2D CNN', color='cornflowerblue')
    axes[1].bar(x_cm + width/2, gnn_cm_vals, width, label='GNN', color='mediumpurple')

    axes[1].set_ylabel('Cell Count')
    axes[1].set_title(f'Confusion Matrix Breakdown [{mode_name.upper()}]', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_cm)
    axes[1].set_xticklabels(cm_labels, fontsize=10)
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5, axis='y')

    for bar in axes[1].patches:
        h = bar.get_height()
        axes[1].annotate(f'{int(h)}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.suptitle(f"Comparative Evaluation [{mode_name.upper()} Power Flow]: 2D CNN vs GNN", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved [{mode_name.upper()}] Model Comparison plot to: {save_path}")
    return cnn, gnn


def compare_cnn_vs_gnn():
    """
    Main comparative function evaluating both DC and AC power flow solver modes.
    """
    # 1. Compare standard/default metrics
    cnn_def = 'data/test_evaluation.json'
    gnn_def = 'data/gnn_test_evaluation.json'
    if os.path.exists(cnn_def) and os.path.exists(gnn_def):
        compare_single_mode(cnn_def, gnn_def, 'Current Solver Mode', 'plots/08_cnn_vs_gnn_comparison.png')

    # 2. Compare DC Mode if files exist
    cnn_dc = 'data/test_evaluation_dc.json'
    gnn_dc = 'data/gnn_test_evaluation_dc.json'
    if os.path.exists(cnn_dc) and os.path.exists(gnn_dc):
        compare_single_mode(cnn_dc, gnn_dc, 'Decoupled DC', 'plots/08_cnn_vs_gnn_comparison_dc.png')

    # 3. Compare AC Mode if files exist
    cnn_ac = 'data/test_evaluation_ac.json'
    gnn_ac = 'data/gnn_test_evaluation_ac.json'
    if os.path.exists(cnn_ac) and os.path.exists(gnn_ac):
        compare_single_mode(cnn_ac, gnn_ac, 'Full AC Contingency', 'plots/09_cnn_vs_gnn_comparison_ac.png')


if __name__ == '__main__':
    compare_cnn_vs_gnn()
