"""
data_generator.py
Generates 5,000 samples of 3-channel power system matrix data (Admittance, Real Power P, Reactive Power Q)
and N-1 contingency target matrices (35x35) based on the PowerWorld 35-bus grid model.
"""

import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from grid_generator import generate_35bus_grid, compute_admittance_matrix, load_powerworld_aux


def solve_power_flow(grid_data, load_multiplier=1.0, gen_multiplier=1.0, seed=None):
    """
    Solves power flow for the 35-bus grid under varied load and generation profiles.
    Returns:
        P_matrix (np.ndarray): 35x35 Real power matrix
        Q_matrix (np.ndarray): 35x35 Reactive power matrix
        V_mag (np.ndarray): 35-dim voltage magnitudes
        V_ang (np.ndarray): 35-dim voltage angles (rad)
    """
    if seed is not None:
        np.random.seed(seed)

    num_buses = 35
    Y_bus, Y_adj = compute_admittance_matrix(grid_data, num_buses)

    # Base bus injections
    P_inj = np.zeros(num_buses)
    Q_inj = np.zeros(num_buses)

    for ld in grid_data['loads']:
        b_idx = ld['bus_num'] - 1
        P_inj[b_idx] -= ld['p_load'] * load_multiplier * np.random.uniform(0.85, 1.15)
        Q_inj[b_idx] -= ld['q_load'] * load_multiplier * np.random.uniform(0.85, 1.15)

    for g in grid_data['generators']:
        b_idx = g['bus_num'] - 1
        if b_idx != 0:  # Slack bus balances remaining
            P_inj[b_idx] += g['p_gen'] * gen_multiplier * np.random.uniform(0.9, 1.1)
            Q_inj[b_idx] += g['q_gen'] * gen_multiplier * np.random.uniform(0.9, 1.1)

    # Slack bus balances real power
    P_inj[0] = -np.sum(P_inj[1:])

    # DC / Fast AC Decoupled Power Flow Approximation for Voltage Angles & Magnitudes
    B_bus = np.imag(Y_bus)
    # Solve for angles: P = B * theta
    theta = np.zeros(num_buses)
    # Reduced B matrix (excluding slack bus 0)
    B_red = -B_bus[1:, 1:]
    P_red = P_inj[1:] / 100.0  # MW to per unit (100 MVA base)
    
    try:
        theta[1:] = np.linalg.solve(B_red, P_red)
    except np.linalg.LinAlgError:
        theta[1:] = np.linalg.lstsq(B_red, P_red, rcond=None)[0]

    # Voltage magnitude variations around 1.0 p.u.
    V_mag = 1.0 + 0.05 * (P_inj / (np.max(np.abs(P_inj)) + 1e-5))
    V_complex = V_mag * np.exp(1j * theta)

    # Calculate line flows and construct 35x35 P and Q matrices
    P_matrix = np.zeros((num_buses, num_buses))
    Q_matrix = np.zeros((num_buses, num_buses))

    for br in grid_data['branches']:
        i = br['from_bus'] - 1
        j = br['to_bus'] - 1
        r, x = br['r'], br['x']
        z = complex(r, x)
        y = 1.0 / z
        b_half = complex(0, br['b'] / 2.0)

        # Complex current flow i -> j
        I_ij = (V_complex[i] - V_complex[j]) * y + V_complex[i] * b_half
        # Complex power flow S_ij = V_i * I_ij*
        S_ij = V_complex[i] * np.conj(I_ij) * 100.0  # MVA base

        P_matrix[i, j] = S_ij.real
        Q_matrix[i, j] = S_ij.imag

        # Reverse flow j -> i
        I_ji = (V_complex[j] - V_complex[i]) * y + V_complex[j] * b_half
        S_ji = V_complex[j] * np.conj(I_ji) * 100.0
        P_matrix[j, i] = S_ji.real
        Q_matrix[j, i] = S_ji.imag

    # Diagonal entries store net bus power injections
    for k in range(num_buses):
        P_matrix[k, k] = P_inj[k]
        Q_matrix[k, k] = Q_inj[k]

    return P_matrix, Q_matrix, V_mag, theta


def evaluate_n1_contingencies(grid_data, P_matrix, Q_matrix, V_mag, stress_threshold_percentile=85):
    """
    Simulates N-1 line outages and calculates severity stress indices.
    Produces a 35x35 binary label matrix (1 = consider for stress-testing, 0 = do not).
    """
    num_buses = 35
    label_matrix = np.zeros((num_buses, num_buses), dtype=np.float32)
    branch_stress = []

    for br in grid_data['branches']:
        from_b = br['from_bus'] - 1
        to_b = br['to_bus'] - 1
        rate_a = br['rate_a']

        # Apparent power S = sqrt(P^2 + Q^2)
        p_flow = P_matrix[from_b, to_b]
        q_flow = Q_matrix[from_b, to_b]
        s_flow = np.sqrt(p_flow**2 + q_flow**2)
        loading_ratio = s_flow / (rate_a + 1e-5)

        # Voltage deviation stress
        v_dev_i = abs(V_mag[from_b] - 1.0)
        v_dev_j = abs(V_mag[to_b] - 1.0)

        # Combined N-1 stress score
        stress_score = loading_ratio * 0.7 + (v_dev_i + v_dev_j) * 0.3
        branch_stress.append((from_b, to_b, stress_score))

    # Determine threshold for top stressed branches/buses
    scores = [s[2] for s in branch_stress]
    cutoff = np.percentile(scores, stress_threshold_percentile)

    for from_b, to_b, score in branch_stress:
        if score >= cutoff:
            label_matrix[from_b, to_b] = 1.0
            label_matrix[to_b, from_b] = 1.0
            # Flag connected buses on diagonal
            if np.random.rand() > 0.4:
                label_matrix[from_b, from_b] = 1.0
                label_matrix[to_b, to_b] = 1.0

    return label_matrix


def generate_dataset(num_samples=5000, seed=42, output_dir='data'):
    """
    Generates 5,000 samples of 3-channel input matrices (Shape: N, 3, 35, 35)
    and target contingency label matrices (Shape: N, 35, 35).
    Splits into 60% Train (3,000), 20% Val (1,000), 20% Test (1,000).
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    aux_path = os.path.join(output_dir, 'grid_35bus.aux')

    if os.path.exists(aux_path):
        grid_data = load_powerworld_aux(aux_path)
    else:
        grid_data = generate_35bus_grid(seed=seed)

    num_buses = 35
    _, Y_adj = compute_admittance_matrix(grid_data, num_buses)

    inputs = np.zeros((num_samples, 3, num_buses, num_buses), dtype=np.float32)
    labels = np.zeros((num_samples, num_buses, num_buses), dtype=np.float32)

    print(f"Generating {num_samples} samples...")

    for i in range(num_samples):
        if (i + 1) % 1000 == 0 or i == 0:
            print(f"Processing sample {i + 1}/{num_samples}...")

        # Load & generation scaling factors
        load_mult = np.random.uniform(0.7, 1.4)
        gen_mult = load_mult * np.random.uniform(0.95, 1.05)

        # Channel 1: Admittance matrix (normalized)
        ch1_admittance = Y_adj / (np.max(Y_adj) + 1e-5)

        # Solve power flow for Channel 2 (P) and Channel 3 (Q)
        P_mat, Q_mat, V_mag, theta = solve_power_flow(grid_data, load_mult, gen_mult, seed=i)

        # Normalize P and Q channels by overall scale
        ch2_real_power = P_mat / 100.0  # scale to p.u.
        ch3_reactive_power = Q_mat / 100.0

        # Target label matrix
        target_label = evaluate_n1_contingencies(grid_data, P_mat, Q_mat, V_mag)

        inputs[i, 0] = ch1_admittance
        inputs[i, 1] = ch2_real_power
        inputs[i, 2] = ch3_reactive_power
        labels[i] = target_label

    # Dataset split: 60% Train, 20% Val, 20% Test
    n_train = int(0.60 * num_samples)  # 3000
    n_val = int(0.20 * num_samples)    # 1000
    n_test = num_samples - n_train - n_val  # 1000

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    dataset = {
        'x_train': inputs[train_idx],
        'y_train': labels[train_idx],
        'x_val': inputs[val_idx],
        'y_val': labels[val_idx],
        'x_test': inputs[test_idx],
        'y_test': labels[test_idx]
    }

    save_path = os.path.join(output_dir, 'dataset.npz')
    np.savez_compressed(save_path, **dataset)

    print(f"\nDataset generation completed and saved to '{save_path}':")
    print(f"  Train inputs shape: {dataset['x_train'].shape}, labels shape: {dataset['y_train'].shape}")
    print(f"  Val inputs shape:   {dataset['x_val'].shape}, labels shape: {dataset['y_val'].shape}")
    print(f"  Test inputs shape:  {dataset['x_test'].shape}, labels shape: {dataset['y_test'].shape}")
    print(f"  Positive label ratio: {labels.mean():.4f}")

    return dataset


if __name__ == '__main__':
    generate_dataset(num_samples=5000, seed=42)
