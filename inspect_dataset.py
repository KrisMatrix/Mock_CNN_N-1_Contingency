"""
inspect_dataset.py
Utility script to load, inspect, and display summaries/slices of dataset.npz
"""

import os
import numpy as np


def inspect_dataset(file_path='data/dataset.npz'):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path)

    print("=" * 60)
    print(f"  DATASET FILE INSPECTION: {file_path}")
    print("=" * 60)
    print(f"Keys inside NPZ file: {list(data.keys())}\n")

    for key in data.keys():
        arr = data[key]
        print(f"Key: '{key}'")
        print(f"  Shape: {arr.shape}")
        print(f"  Data type: {arr.dtype}")
        print(f"  Min value: {arr.min():.4f}")
        print(f"  Max value: {arr.max():.4f}")
        print(f"  Mean value: {arr.mean():.4f}")
        if 'y' in key:
            print(f"  Positive entries (1s): {int(arr.sum())} / {arr.size} ({arr.mean()*100:.2f}%)")
        print("-" * 60)

    # Inspect Sample 0 in Train Set
    x_sample = data['x_train'][0]  # Shape (3, 35, 35)
    y_sample = data['y_train'][0]  # Shape (35, 35)

    print("\nSample 0 Inspection:")
    print(f"  Channel 1 (Admittance |Y_bus|) shape: {x_sample[0].shape}, Max: {x_sample[0].max():.4f}")
    print(f"  Channel 2 (Real Power P)       shape: {x_sample[1].shape}, Max: {x_sample[1].max():.4f}")
    print(f"  Channel 3 (Reactive Power Q)   shape: {x_sample[2].shape}, Max: {x_sample[2].max():.4f}")
    print(f"  Target Label Matrix Y          shape: {y_sample.shape},  Total 1s: {int(y_sample.sum())}")

    print("\nFirst 5x5 submatrix slice of Channel 1 (Admittance):")
    print(np.round(x_sample[0][:5, :5], 4))

    print("\nFirst 5x5 submatrix slice of Target Label Matrix Y:")
    print(y_sample[:5, :5].astype(int))
    print("=" * 60)


if __name__ == '__main__':
    inspect_dataset()
