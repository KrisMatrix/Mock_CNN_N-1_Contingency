"""
run_pipeline.py
Python Master Orchestrator for N-1 Contingency Recommendation System.
Integrates complete workflow for both 2D CNN and Graph Neural Network (GNN) models
across Decoupled DC Powerflow and Full AC Newton-Raphson Contingency Solvers.
"""

import os
import sys
import argparse
import time

from grid_generator import generate_35bus_grid, export_powerworld_aux
from data_generator import generate_dataset
from train import train_cnn_model
from evaluate import evaluate_cnn_model
from visualize import visualize_results
from train_gnn import train_gnn_model
from evaluate_gnn import evaluate_gnn_model
from visualize_gnn import visualize_gnn_results
from compare_models import compare_cnn_vs_gnn


def run_pipeline_for_method(
    method='ac',
    num_samples=5000,
    epochs=25,
    batch_size=32,
    lr=1e-3,
    model_choice='all',
    skip_data_gen=False
):
    print("=" * 75)
    print(f"  RUNNING N-1 PIPELINE [{method.upper()} POWER FLOW SOLVER]")
    print("=" * 75)

    # 1. Grid Model
    grid = generate_35bus_grid(seed=42)
    export_powerworld_aux(grid, os.path.join('data', 'grid_35bus.aux'))

    # 2. Dataset Generation
    dataset_path = os.path.join('data', 'dataset.npz')
    if skip_data_gen and os.path.exists(dataset_path):
        print(f"\nSkipping [{method.upper()}] Dataset Generation. Using existing '{dataset_path}'.")
    else:
        print(f"\nGenerating Shared {num_samples}-Sample Dataset with [{method.upper()}] Solver...")
        generate_dataset(num_samples=num_samples, seed=42, method=method)

    # 3. 2D CNN Pipeline
    if model_choice in ['cnn', 'all']:
        print(f"\nTraining 2D CNN Model on [{method.upper()}] Dataset ({epochs} epochs)...")
        train_cnn_model(epochs=epochs, batch_size=batch_size, lr=lr)
        evaluate_cnn_model()
        visualize_results()

    # 4. GNN Pipeline
    if model_choice in ['gnn', 'all']:
        print(f"\nTraining Graph Neural Network (GNN) Model on [{method.upper()}] Dataset ({epochs} epochs)...")
        train_gnn_model(epochs=epochs, batch_size=batch_size, lr=lr)
        evaluate_gnn_model()
        visualize_gnn_results()

    # 5. Model Comparison
    if model_choice == 'all':
        compare_cnn_vs_gnn()


def main():
    parser = argparse.ArgumentParser(
        description="Python Master Orchestrator for N-1 Contingency System (CNN & GNN across DC and AC Solvers)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--method', type=str, default='ac', choices=['dc', 'ac', 'all'],
                        help="Power flow solver method ('dc', 'ac', or 'all')")
    parser.add_argument('--samples', type=int, default=5000, help="Number of dataset samples to generate")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs per model")
    parser.add_argument('--batch-size', type=int, default=32, help="Training batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Optimizer initial learning rate")
    parser.add_argument('--model', type=str, default='all', choices=['cnn', 'gnn', 'all'],
                        help="Specify model workflow to run ('cnn', 'gnn', or 'all')")
    parser.add_argument('--skip-data-gen', action='store_true',
                        help="Skip dataset generation if data/dataset.npz already exists")

    args = parser.parse_args()
    start_time = time.time()

    if args.method in ['dc', 'all']:
        run_pipeline_for_method('dc', args.samples, args.epochs, args.batch_size, args.lr, args.model, args.skip_data_gen)

    if args.method in ['ac', 'all']:
        run_pipeline_for_method('ac', args.samples, args.epochs, args.batch_size, args.lr, args.model, args.skip_data_gen)

    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)

    print("\n" + "=" * 75)
    print("  PYTHON MASTER ORCHESTRATOR COMPLETED SUCCESSFULLY!")
    print(f"  Total Execution Time: {int(mins)}m {secs:.1f}s")
    print("  Shared Dataset: data/dataset.npz")
    print("  CNN Metrics:    data/test_evaluation.json")
    print("  GNN Metrics:    data/gnn_test_evaluation.json")
    print("  All Plots:      plots/")
    print("=" * 75)


if __name__ == '__main__':
    main()
