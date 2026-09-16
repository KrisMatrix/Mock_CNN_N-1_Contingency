"""
evaluate.py
Evaluation engine for N-1 Contingency Recommendation CNN on the 1,000 un-seen test set samples.
Computes Test Loss, Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, and Confusion Matrix.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix
from model import load_dataset
from train import weighted_bce_loss


def evaluate_cnn_model(model_path='checkpoints/best_model.keras', dataset_path='data/dataset.npz', output_json='data/test_evaluation.json', pos_weight=15.0):
    """
    Evaluates the trained ContingencyCNN model on the 1,000 test set samples.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # 1. Load Data
    _, _, (x_test, y_test) = load_dataset(dataset_path)

    # 2. Load Trained Model
    loss_fn = weighted_bce_loss(pos_weight=pos_weight)
    model = keras.models.load_model(model_path, custom_objects={'loss_fn': loss_fn, 'F1ScoreMetric': keras.metrics.Metric})

    print(f"\nRunning Model Inference on 1,000 Test Samples...")
    logits = model.predict(x_test, batch_size=32, verbose=1)  # shape: (1000, 35, 35)

    # Convert logits to probabilities via Sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Flatten tensors for scikit-learn metrics calculation
    y_true_flat = y_test.flatten()
    y_prob_flat = probs.flatten()
    
    # Binary predictions using 0.5 probability threshold (logit > 0)
    y_pred_flat = (y_prob_flat >= 0.5).astype(np.float32)

    # 3. Calculate Comprehensive Metrics
    test_loss = float(loss_fn(tf.constant(y_test), tf.constant(logits)).numpy())
    acc = float(accuracy_score(y_true_flat, y_pred_flat))
    prec = float(precision_score(y_true_flat, y_pred_flat, zero_division=0))
    rec = float(recall_score(y_true_flat, y_pred_flat, zero_division=0))
    f1 = float(f1_score(y_true_flat, y_pred_flat, zero_division=0))
    roc_auc = float(roc_auc_score(y_true_flat, y_prob_flat))

    cm = confusion_matrix(y_true_flat, y_pred_flat)
    tn, fp, fn, tp = cm.ravel()

    results = {
        'test_samples': int(len(x_test)),
        'total_matrix_cells': int(len(y_true_flat)),
        'positive_labels_count': int(y_true_flat.sum()),
        'test_loss': round(test_loss, 4),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    }

    # Save Evaluation Results
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION RESULTS (1,000 SAMPLES)")
    print("=" * 60)
    print(f"  Test Loss:          {test_loss:.4f}")
    print(f"  Accuracy:           {acc * 100:.2f}%")
    print(f"  Precision:          {prec * 100:.2f}%")
    print(f"  Recall:             {rec * 100:.2f}%")
    print(f"  F1-Score:           {f1:.4f}")
    print(f"  ROC-AUC Score:      {roc_auc:.4f}")
    print("-" * 60)
    print(f"  Confusion Matrix:   TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("=" * 60)
    print(f"Saved test evaluation metrics to: {output_json}")

    return results, probs, y_test


if __name__ == '__main__':
    evaluate_cnn_model()
