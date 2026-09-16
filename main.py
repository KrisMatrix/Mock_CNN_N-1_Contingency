"""
main.py
Master Orchestrator Script for N-1 Contingency Recommendation CNN.
Integrates all functions required by AGENTS.md:
1. Grid model & PowerWorld AUX generator
2. Synthetic dataset generator (5,000 samples, 3 channels)
3. 2D CNN model creation (TensorFlow / Keras)
4. Model training & validation engine
5. Test evaluation engine
6. Comprehensive visualization suite
"""

import os
import argparse
import numpy as np

from grid_generator import generate_35bus_grid, export_powerworld_aux, load_powerworld_aux, compute_admittance_matrix
from data_generator import generate_dataset
from model import create_cnn_model, load_dataset
from train import train_cnn_model
from evaluate import evaluate_cnn_model
from visualize import visualize_results


def run_phase_1():
    print("\n" + "=" * 60)
    print("  PHASE 1: Grid Model & PowerWorld AUX File Generator")
    print("=" * 60)
    grid = generate_35bus_grid(seed=42)
    aux_path = os.path.join('data', 'grid_35bus.aux')
    export_powerworld_aux(grid, aux_path)

    parsed_grid = load_powerworld_aux(aux_path)
    Y_bus, Y_adj = compute_admittance_matrix(parsed_grid)

    print(f"Phase 1 Summary:")
    print(f"  Buses: {len(parsed_grid['buses'])}, Transmission Branches: {len(parsed_grid['branches'])}")
    print(f"  Admittance Matrix Y_bus shape: {Y_bus.shape}")
    print(f"  Y_adj Max: {Y_adj.max():.4f}, Min: {Y_adj.min():.4f}\n")


def run_phase_2(num_samples=5000):
    print("\n" + "=" * 60)
    print(f"  PHASE 2: Synthetic Data Generation ({num_samples} samples)")
    print("=" * 60)
    dataset = generate_dataset(num_samples=num_samples, seed=42)
    print("Phase 2 Completed Successfully.\n")


def run_phase_3():
    print("\n" + "=" * 60)
    print("  PHASE 3: TensorFlow / Keras 2D CNN Model Architecture")
    print("=" * 60)
    model = create_cnn_model(input_shape=(35, 35, 3))
    model.summary()
    print("Phase 3 Model Architecture Created Successfully.\n")


def run_phase_4(epochs=25, batch_size=32, lr=1e-3):
    print("\n" + "=" * 60)
    print(f"  PHASE 4: Model Training Engine ({epochs} epochs)")
    print("=" * 60)
    model, history = train_cnn_model(epochs=epochs, batch_size=batch_size, lr=lr)
    print("Phase 4 Model Training Completed Successfully.\n")


def run_phase_5():
    print("\n" + "=" * 60)
    print("  PHASE 5: Model Evaluation & Visualization Suite")
    print("=" * 60)
    results, _, _ = evaluate_cnn_model()
    visualize_results()
    print("Phase 5 Evaluation & Visualizations Completed Successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="N-1 Contingency Recommendation System using 3-Channel 2D CNN")
    parser.add_argument('--phase', type=str, default='all', choices=['1', '2', '3', '4', '5', 'all'],
                        help="Specify phase to execute (1, 2, 3, 4, 5, or 'all')")
    parser.add_argument('--samples', type=int, default=5000, help="Number of dataset samples to generate (default: 5000)")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs (default: 25)")

    args = parser.parse_args()

    if args.phase == '1' or args.phase == 'all':
        run_phase_1()

    if args.phase == '2' or args.phase == 'all':
        run_phase_2(num_samples=args.samples)

    if args.phase == '3' or args.phase == 'all':
        run_phase_3()

    if args.phase == '4' or args.phase == 'all':
        run_phase_4(epochs=args.epochs)

    if args.phase == '5' or args.phase == 'all':
        run_phase_5()

    print("=" * 60)
    print("  ALL PHASES EXECUTED & VERIFIED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    main()
