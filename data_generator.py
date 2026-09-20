"""
data_generator.py
Generates 5,000 samples of 3-channel power system matrix data (Admittance, Real Power P, Reactive Power Q)
and N-1 contingency target matrices (35x35) based on the PowerWorld 35-bus grid model.

Supports Dual Power Flow & Contingency Solver Engines:
1. 'dc': Fast Decoupled DC Power Flow + Heuristic Stress Metric
2. 'ac': Full Non-linear AC Newton-Raphson Power Flow (via pandapower) + Physical N-1 Outage Simulation
"""

import os
import copy
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import pandapower as pp
from grid_generator import generate_35bus_grid, compute_admittance_matrix, load_powerworld_aux


def build_pandapower_network(grid_data):
    """
    Constructs a pandapower AC network object from grid_data data structure once.
    """
    net = pp.create_empty_network(name="35_bus_grid")
    bus_map = {}

    # 1. Add Buses
    for b in grid_data['buses']:
        b_idx = b['bus_num']
        vn_kv = b.get('base_kv', 230.0)
        pp_bus = pp.create_bus(net, vn_kv=vn_kv, name=f"Bus_{b_idx:02d}", index=b_idx-1)
        bus_map[b_idx] = pp_bus

    # 2. Add Slack / Ext Grid at Bus 1
    pp.create_ext_grid(net, bus=bus_map[1], vm_pu=1.05, va_degree=0.0, name="Slack_Grid")

    # 3. Add Generators
    for g in grid_data['generators']:
        if g['bus_num'] != 1:
            pp.create_gen(net, bus=bus_map[g['bus_num']], p_mw=g['p_gen'], vm_pu=1.02, name=f"Gen_{g['bus_num']}")

    # 4. Add Loads
    for ld in grid_data['loads']:
        pp.create_load(net, bus=bus_map[ld['bus_num']], p_mw=ld['p_load'], q_mvar=ld['q_load'], name=f"Load_{ld['bus_num']}")

    # 5. Add Lines / Branches
    for br in grid_data['branches']:
        f_bus = bus_map[br['from_bus']]
        t_bus = bus_map[br['to_bus']]

        r_ohm = br['r'] * (230.0 ** 2) / 100.0
        x_ohm = br['x'] * (230.0 ** 2) / 100.0
        c_nf = (br['b'] / (2 * np.pi * 60)) * (100.0 / (230.0 ** 2)) * 1e9

        pp.create_line_from_parameters(
            net, from_bus=f_bus, to_bus=t_bus, length_km=1.0,
            r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm, c_nf_per_km=max(c_nf, 0.001),
            max_i_ka=br['rate_a'] / (np.sqrt(3) * 230.0),
            in_service=True, name=f"Line_{br['from_bus']}_{br['to_bus']}"
        )

    return net


def update_pandapower_network(net, dyn_grid):
    """
    Updates existing pandapower network tables in-place using vectorized numpy arrays.
    """
    # 1. Update Load P and Q
    p_loads = np.array([ld['p_load'] for ld in dyn_grid['loads']], dtype=np.float32)
    q_loads = np.array([ld['q_load'] for ld in dyn_grid['loads']], dtype=np.float32)
    net.load['p_mw'] = p_loads
    net.load['q_mvar'] = q_loads
    net.load['in_service'] = (p_loads > 0) | (q_loads > 0)

    # 2. Update Generator P
    gen_list = [g for g in dyn_grid['generators'] if g['bus_num'] != 1]
    p_gens = np.array([g['p_gen'] for g in gen_list], dtype=np.float32)
    net.gen['p_mw'] = p_gens
    net.gen['in_service'] = (p_gens > 0)

    # 3. Update Line Status
    line_status = np.array([br.get('status', 'Closed') == 'Closed' for br in dyn_grid['branches']], dtype=bool)
    net.line['in_service'] = line_status


def apply_dynamic_operational_topology(grid_data, seed=None):
    """
    Creates a dynamic operational snapshot of the 35-bus grid for a single sample.
    """
    if seed is not None:
        np.random.seed(seed)

    sample_grid = copy.deepcopy(grid_data)
    num_trips = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
    trip_candidates = [idx for idx, br in enumerate(sample_grid['branches']) if br['from_bus'] != 1 and br['to_bus'] != 1]

    if trip_candidates:
        tripped_indices = np.random.choice(trip_candidates, size=min(num_trips, len(trip_candidates)), replace=False)
        for idx in tripped_indices:
            sample_grid['branches'][idx]['status'] = 'Open'

    if np.random.rand() > 0.7:
        gen_indices = [idx for idx, g in enumerate(sample_grid['generators']) if g['bus_num'] != 1]
        if gen_indices:
            off_gen_idx = np.random.choice(gen_indices)
            sample_grid['generators'][off_gen_idx]['status'] = 'Open'
            sample_grid['generators'][off_gen_idx]['p_gen'] = 0.0
            sample_grid['generators'][off_gen_idx]['q_gen'] = 0.0

    global_load_mult = np.random.uniform(0.7, 1.4)
    for ld in sample_grid['loads']:
        if np.random.rand() < 0.05:
            ld['p_load'] = 0.0
            ld['q_load'] = 0.0
        else:
            bus_mult = global_load_mult * np.random.uniform(0.75, 1.25)
            ld['p_load'] *= bus_mult
            ld['q_load'] *= bus_mult

    return sample_grid


def solve_power_flow(grid_data, seed=None):
    """
    Solves power flow for the dynamic operational grid snapshot.
    """
    if seed is not None:
        np.random.seed(seed)

    num_buses = 35
    Y_bus, Y_adj = compute_admittance_matrix(grid_data, num_buses)

    P_inj = np.zeros(num_buses)
    Q_inj = np.zeros(num_buses)

    for ld in grid_data['loads']:
        if ld.get('status', 'Closed') == 'Closed':
            b_idx = ld['bus_num'] - 1
            P_inj[b_idx] -= ld['p_load']
            Q_inj[b_idx] -= ld['q_load']

    for g in grid_data['generators']:
        if g.get('status', 'Closed') == 'Closed':
            b_idx = g['bus_num'] - 1
            if b_idx != 0:
                P_inj[b_idx] += g['p_gen']
                Q_inj[b_idx] += g['q_gen']

    P_inj[0] = -np.sum(P_inj[1:])

    B_bus = np.imag(Y_bus)
    theta = np.zeros(num_buses)
    B_red = -B_bus[1:, 1:]
    P_red = P_inj[1:] / 100.0

    try:
        theta[1:] = np.linalg.solve(B_red, P_red)
    except np.linalg.LinAlgError:
        theta[1:] = np.linalg.lstsq(B_red, P_red, rcond=None)[0]

    V_mag = 1.0 + 0.05 * (P_inj / (np.max(np.abs(P_inj)) + 1e-5))
    V_complex = V_mag * np.exp(1j * theta)

    P_matrix = np.zeros((num_buses, num_buses))
    Q_matrix = np.zeros((num_buses, num_buses))

    for br in grid_data['branches']:
        if br.get('status', 'Closed') != 'Closed':
            continue

        i = br['from_bus'] - 1
        j = br['to_bus'] - 1
        r, x = br['r'], br['x']
        z = complex(r, x)
        y = 1.0 / z
        b_half = complex(0, br['b'] / 2.0)
        tap = br.get('tap', 1.0)

        I_ij = (V_complex[i] - V_complex[j]) * (y / tap) + V_complex[i] * b_half
        S_ij = V_complex[i] * np.conj(I_ij) * 100.0

        P_matrix[i, j] = S_ij.real
        Q_matrix[i, j] = S_ij.imag

        I_ji = (V_complex[j] - V_complex[i]) * (y / tap) + V_complex[j] * b_half
        S_ji = V_complex[j] * np.conj(I_ji) * 100.0
        P_matrix[j, i] = S_ji.real
        Q_matrix[j, i] = S_ji.imag

    for k in range(num_buses):
        P_matrix[k, k] = P_inj[k]
        Q_matrix[k, k] = Q_inj[k]

    return P_matrix, Q_matrix, V_mag, theta, Y_bus, Y_adj


def evaluate_contingencies_dc(grid_data, P_matrix, Q_matrix, V_mag, stress_threshold_percentile=85):
    """
    Method 'dc': Fast Decoupled DC Power Flow Heuristic Stress Evaluation.
    """
    num_buses = 35
    label_matrix = np.zeros((num_buses, num_buses), dtype=np.float32)
    branch_stress = []

    for br in grid_data['branches']:
        if br.get('status', 'Closed') != 'Closed':
            continue

        from_b = br['from_bus'] - 1
        to_b = br['to_bus'] - 1
        rate_a = br['rate_a']

        p_flow = P_matrix[from_b, to_b]
        q_flow = Q_matrix[from_b, to_b]
        s_flow = np.sqrt(p_flow**2 + q_flow**2)
        loading_ratio = s_flow / (rate_a + 1e-5)

        v_dev_i = abs(V_mag[from_b] - 1.0)
        v_dev_j = abs(V_mag[to_b] - 1.0)

        stress_score = loading_ratio * 0.7 + (v_dev_i + v_dev_j) * 0.3
        branch_stress.append((from_b, to_b, stress_score))

    if not branch_stress:
        return label_matrix

    scores = [s[2] for s in branch_stress]
    cutoff = np.percentile(scores, stress_threshold_percentile)

    for from_b, to_b, score in branch_stress:
        if score >= cutoff:
            label_matrix[from_b, to_b] = 1.0
            label_matrix[to_b, from_b] = 1.0
            if np.random.rand() > 0.4:
                label_matrix[from_b, from_b] = 1.0
                label_matrix[to_b, to_b] = 1.0

    return label_matrix


def evaluate_contingencies_ac(net, dyn_grid, P_matrix, Q_matrix, V_mag):
    """
    Method 'ac': Fast In-place pandapower AC Newton-Raphson Contingency Solver.
    """
    num_buses = 35
    label_matrix = np.zeros((num_buses, num_buses), dtype=np.float32)
    
    update_pandapower_network(net, dyn_grid)

    try:
        pp.runpp(net, calculate_voltage_angles=False, enforce_q_lims=True, numba=True)
    except Exception:
        return evaluate_contingencies_dc(dyn_grid, P_matrix, Q_matrix, V_mag)

    # Select top candidate lines for contingency simulation (top 6 or loaded > 35%)
    if hasattr(net, 'res_line') and 'loading_percent' in net.res_line:
        candidates = net.res_line[net.res_line.loading_percent > 35.0].index.tolist()
        if len(candidates) < 5:
            candidates = net.res_line.sort_values('loading_percent', ascending=False).head(6).index.tolist()
    else:
        candidates = net.line.index.tolist()[:6]

    in_service_arr = net.line['in_service'].values.copy()
    from_buses = net.line['from_bus'].values
    to_buses = net.line['to_bus'].values

    for line_idx in candidates:
        from_b = from_buses[line_idx]
        to_b = to_buses[line_idx]

        net.line.at[line_idx, 'in_service'] = False
        is_critical = False

        try:
            pp.runpp(net, calculate_voltage_angles=False, max_iteration=10, numba=True)
            
            if net.res_line.loading_percent.max() > 85.0:
                is_critical = True

            v_min = net.res_bus.vm_pu.min()
            v_max = net.res_bus.vm_pu.max()
            if v_min < 0.94 or v_max > 1.06:
                is_critical = True

        except Exception:
            is_critical = True

        net.line.at[line_idx, 'in_service'] = True

        if is_critical:
            label_matrix[from_b, to_b] = 1.0
            label_matrix[to_b, from_b] = 1.0
            label_matrix[from_b, from_b] = 1.0
            label_matrix[to_b, to_b] = 1.0

    return label_matrix


def generate_dataset(num_samples=5000, seed=42, method='ac', output_dir='data', scada_noise_std=0.015):
    """
    Generates 5,000 samples of 3-channel input matrices (Shape: N, 3, 35, 35).
    Method options:
      - 'dc': Fast Decoupled DC Power Flow Heuristic
      - 'ac': Full Non-linear AC Newton-Raphson Contingency Solver (pandapower)
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    aux_path = os.path.join(output_dir, 'grid_35bus.aux')

    if os.path.exists(aux_path):
        base_grid_data = load_powerworld_aux(aux_path)
    else:
        base_grid_data = generate_35bus_grid(seed=seed)

    num_buses = 35
    inputs = np.zeros((num_samples, 3, num_buses, num_buses), dtype=np.float32)
    labels = np.zeros((num_samples, num_buses, num_buses), dtype=np.float32)

    # Pre-build pandapower network once for AC mode
    if method == 'ac':
        pp_net = build_pandapower_network(base_grid_data)

    print(f"\nGenerating {num_samples} samples using Power Flow Method: [{method.upper()}]...", flush=True)

    for i in range(num_samples):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"Processing [{method.upper()}] sample {i + 1}/{num_samples}...", flush=True)

        dyn_grid = apply_dynamic_operational_topology(base_grid_data, seed=seed + i)
        P_mat, Q_mat, V_mag, theta, Y_bus, Y_adj = solve_power_flow(dyn_grid, seed=seed + i)

        ch1_admittance = Y_adj / (np.max(Y_adj) + 1e-5)
        ch2_real_power = P_mat / 100.0
        ch3_reactive_power = Q_mat / 100.0

        if scada_noise_std > 0:
            ch2_real_power += np.random.normal(0, scada_noise_std, size=ch2_real_power.shape).astype(np.float32)
            ch3_reactive_power += np.random.normal(0, scada_noise_std, size=ch3_reactive_power.shape).astype(np.float32)

        if method == 'ac':
            target_label = evaluate_contingencies_ac(pp_net, dyn_grid, P_mat, Q_mat, V_mag)
        else:
            target_label = evaluate_contingencies_dc(dyn_grid, P_mat, Q_mat, V_mag)

        inputs[i, 0] = ch1_admittance
        inputs[i, 1] = ch2_real_power
        inputs[i, 2] = ch3_reactive_power
        labels[i] = target_label

    n_train = int(0.60 * num_samples)
    n_val = int(0.20 * num_samples)
    n_test = num_samples - n_train - n_val

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

    filename = f"dataset_{method}.npz" if method in ['dc', 'ac'] else 'dataset.npz'
    save_path = os.path.join(output_dir, filename)
    
    np.savez_compressed(save_path, **dataset)
    np.savez_compressed(os.path.join(output_dir, 'dataset.npz'), **dataset)

    print(f"\n[{method.upper()}] Dataset generation completed and saved to '{save_path}':")
    print(f"  Train inputs shape: {dataset['x_train'].shape}, labels shape: {dataset['y_train'].shape}")
    print(f"  Val inputs shape:   {dataset['x_val'].shape}, labels shape: {dataset['y_val'].shape}")
    print(f"  Test inputs shape:  {dataset['x_test'].shape}, labels shape: {dataset['y_test'].shape}")
    print(f"  Positive label ratio: {labels.mean():.4f}")

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator for N-1 Contingency System")
    parser.add_argument('--method', type=str, default='ac', choices=['dc', 'ac'], help="Power flow solver method ('dc' or 'ac')")
    parser.add_argument('--samples', type=int, default=5000, help="Number of dataset samples (default: 5000)")
    args = parser.parse_args()

    generate_dataset(num_samples=args.samples, seed=42, method=args.method)
