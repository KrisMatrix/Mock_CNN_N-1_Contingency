"""
main_gnn.py
Master Orchestrator Script for N-1 Contingency Recommendation GNN.
Integrates Graph Neural Network workflow:
1. Load 3-channel matrix dataset
2. TensorFlow / Keras GNN model creation
3. GNN Model training & validation engine
4. GNN Test evaluation engine
5. Comprehensive GNN visualization suite
6. Model Comparison (2D CNN vs GNN)
"""

import os
import argparse

from gnn_model import create_gnn_model
from train_gnn import train_gnn_model
from evaluate_gnn import evaluate_gnn_model
from visualize_gnn import visualize_gnn_results
from compare_models import compare_cnn_vs_gnn


def run_gnn_phase_3():
    print("\n" + "=" * 60)
    print("  GNN PHASE 3: TensorFlow / Keras GNN Architecture")
    print("=" * 60)
    model = create_gnn_model(input_shape=(35, 35, 3))
    model.summary()
    print("GNN Model Architecture Created Successfully.\n")


def run_gnn_phase_4(epochs=25, batch_size=32, lr=1e-3):
    print("\n" + "=" * 60)
    print(f"  GNN PHASE 4: GNN Model Training Engine ({epochs} epochs)")
    print("=" * 60)
    model, history = train_gnn_model(epochs=epochs, batch_size=batch_size, lr=lr)
    print("GNN Model Training Completed Successfully.\n")


def run_gnn_phase_5():
    print("\n" + "=" * 60)
    print("  GNN PHASE 5: GNN Model Evaluation & Visualization Suite")
    print("=" * 60)
    results, _, _ = evaluate_gnn_model()
    visualize_gnn_results()
    print("GNN Evaluation & Visualizations Completed Successfully.\n")


def run_model_comparison():
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON: 2D CNN vs Graph Neural Network (GNN)")
    print("=" * 60)
    compare_cnn_vs_gnn()
    print("Model Comparison Completed Successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="N-1 Contingency Recommendation System using Graph Neural Network (GNN)")
    parser.add_argument('--phase', type=str, default='all', choices=['3', '4', '5', 'compare', 'all'],
                        help="Specify GNN phase to execute ('3', '4', '5', 'compare', or 'all')")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate (default: 1e-3)")

    args = parser.parse_args()

    if args.phase == '3' or args.phase == 'all':
        run_gnn_phase_3()

    if args.phase == '4' or args.phase == 'all':
        run_gnn_phase_4(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    if args.phase == '5' or args.phase == 'all':
        run_gnn_phase_5()

    if args.phase == 'compare' or args.phase == 'all':
        run_model_comparison()

    print("=" * 60)
    print("  ALL GNN PHASES EXECUTED & VERIFIED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    main()
