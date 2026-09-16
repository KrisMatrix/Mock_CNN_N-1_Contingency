"""
grid_generator.py
Module for generating a synthetic 35-bus power system model, exporting/importing
PowerWorld AUX files, and computing the admittance-adjacency matrix (Y_bus magnitude).
"""

import os
import re
import numpy as np
import pandas as pd


def generate_35bus_grid(seed=42):
    """
    Generates synthetic 35-bus power grid data structures:
    - Buses (1 to 35)
    - Generators (6 generators at major supply buses)
    - Loads (25 load demand centers)
    - Branches (52 transmission lines & transformers)

    Returns:
        dict: Containing 'buses', 'generators', 'loads', 'branches' data.
    """
    np.random.seed(seed)
    num_buses = 35

    # 1. Buses definition
    # Bus Types: 1 = Slack, 2 = PV (Generator), 3 = PQ (Load)
    buses = []
    slack_bus = 1
    gen_buses = [2, 5, 10, 18, 25, 32]
    
    for bus_id in range(1, num_buses + 1):
        if bus_id == slack_bus:
            b_type = 1  # Slack
            v_mag = 1.05
            v_ang = 0.0
            base_kv = 230.0
        elif bus_id in gen_buses:
            b_type = 2  # PV
            v_mag = 1.02 + np.random.uniform(-0.01, 0.02)
            v_ang = np.random.uniform(-5.0, 5.0)
            base_kv = 230.0
        else:
            b_type = 3  # PQ
            v_mag = 1.00 + np.random.uniform(-0.03, 0.02)
            v_ang = np.random.uniform(-10.0, 2.0)
            base_kv = 138.0 if bus_id > 20 else 230.0

        buses.append({
            'bus_num': bus_id,
            'bus_name': f"Bus_{bus_id:02d}",
            'base_kv': base_kv,
            'bus_type': b_type,
            'v_mag': round(float(v_mag), 4),
            'v_ang': round(float(v_ang), 2)
        })

    # 2. Generators definition
    all_gen_buses = [slack_bus] + gen_buses
    generators = []
    for idx, bus_id in enumerate(all_gen_buses):
        p_max = 300.0 if bus_id in [1, 2] else 180.0
        p_gen = p_max * np.random.uniform(0.5, 0.85) if bus_id != 1 else 250.0
        q_gen = p_gen * np.random.uniform(0.1, 0.3)
        
        generators.append({
            'bus_num': bus_id,
            'gen_id': "1",
            'p_gen': round(float(p_gen), 2),
            'q_gen': round(float(q_gen), 2),
            'p_max': float(p_max),
            'p_min': 0.0,
            'q_max': round(float(p_max * 0.6), 2),
            'q_min': round(float(-p_max * 0.3), 2),
            'status': "Closed"
        })

    # 3. Loads definition (distributed across PQ buses and some PV buses)
    loads = []
    load_buses = [b for b in range(1, num_buses + 1) if b not in [1, 2]]
    for bus_id in load_buses:
        p_load = np.random.uniform(15.0, 65.0)
        power_factor = np.random.uniform(0.88, 0.96)
        q_load = p_load * np.tan(np.arccos(power_factor))
        
        loads.append({
            'bus_num': bus_id,
            'load_id': "1",
            'p_load': round(float(p_load), 2),
            'q_load': round(float(q_load), 2),
            'status': "Closed"
        })

    # 4. Branches (Transmission Lines & Transformers) definition
    # Create a 35-bus connected graph (ring main + meshed grid cross-links)
    branch_edges = []
    # Primary ring backbone
    for i in range(1, num_buses):
        branch_edges.append((i, i + 1))
    branch_edges.append((num_buses, 1))

    # Cross-grid interconnections for realism and mesh redundancy
    extra_edges = [
        (1, 7), (1, 12), (2, 9), (3, 15), (4, 18), (5, 22), (6, 25),
        (8, 20), (10, 28), (11, 31), (13, 27), (14, 33), (16, 35), (17, 30),
        (19, 26), (21, 34), (23, 32), (24, 29), (5, 14), (9, 25)
    ]
    branch_edges.extend(extra_edges)

    branches = []
    for from_b, to_b in branch_edges:
        r = np.random.uniform(0.005, 0.04)
        x = r * np.random.uniform(3.5, 8.0)  # X/R ratio between 3.5 and 8.0
        b = np.random.uniform(0.005, 0.03)   # Shunt line charging
        is_xfmr = (from_b in gen_buses or to_b in gen_buses) and np.random.rand() > 0.6
        tap = round(float(np.random.uniform(0.97, 1.03)), 4) if is_xfmr else 1.0
        rate_a = round(float(np.random.uniform(150.0, 400.0)), 1)

        branches.append({
            'from_bus': from_b,
            'to_bus': to_b,
            'circuit_id': "1",
            'r': round(float(r), 5),
            'x': round(float(x), 5),
            'b': round(float(b), 5),
            'tap': tap,
            'status': "Closed",
            'rate_a': rate_a
        })

    return {
        'buses': buses,
        'generators': generators,
        'loads': loads,
        'branches': branches
    }


def export_powerworld_aux(grid_data, file_path):
    """
    Exports grid data into a PowerWorld Auxiliary (.aux) file format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write("// PowerWorld Auxiliary File\n")
        f.write("// 35-Bus Power System Model for Contingency Recommendation CNN\n\n")

        # BUS Data
        f.write("DATA (BUS, [BusNum, BusName, BusKV, BusType, Vang, Vmag])\n{\n")
        for b in grid_data['buses']:
            b_type_str = "Slack" if b['bus_type'] == 1 else ("PV" if b['bus_type'] == 2 else "PQ")
            f.write(f'  {b["bus_num"]} "{b["bus_name"]}" {b["base_kv"]:.1f} "{b_type_str}" {b["v_ang"]:.2f} {b["v_mag"]:.4f}\n')
        f.write("}\n\n")

        # GEN Data
        f.write("DATA (GEN, [BusNum, GenID, GenMW, GenMVAR, GenMWMax, GenMWMin, GenMVARMax, GenMVARMin, GenStatus])\n{\n")
        for g in grid_data['generators']:
            f.write(f'  {g["bus_num"]} "{g["gen_id"]}" {g["p_gen"]:.2f} {g["q_gen"]:.2f} {g["p_max"]:.1f} {g["p_min"]:.1f} {g["q_max"]:.1f} {g["q_min"]:.1f} "{g["status"]}"\n')
        f.write("}\n\n")

        # LOAD Data
        f.write("DATA (LOAD, [BusNum, LoadID, LoadMW, LoadMVAR, LoadStatus])\n{\n")
        for ld in grid_data['loads']:
            f.write(f'  {ld["bus_num"]} "{ld["load_id"]}" {ld["p_load"]:.2f} {ld["q_load"]:.2f} "{ld["status"]}"\n')
        f.write("}\n\n")

        # BRANCH Data
        f.write("DATA (BRANCH, [BusNum, BusNum:1, Circuit, LineR, LineX, LineC, LineStatus, RateA])\n{\n")
        for br in grid_data['branches']:
            f.write(f'  {br["from_bus"]} {br["to_bus"]} "{br["circuit_id"]}" {br["r"]:.5f} {br["x"]:.5f} {br["b"]:.5f} "{br["status"]}" {br["rate_a"]:.1f}\n')
        f.write("}\n\n")

    print(f"Successfully exported PowerWorld AUX file to: {file_path}")


def _parse_aux_line(line):
    """
    Utility parser for PowerWorld AUX line tokens (handles quoted strings, floats, ints).
    """
    pattern = r'"([^"]*)"|(-?\d+(?:\.\d+)?)|(\S+)'
    tokens = []
    for match in re.finditer(pattern, line):
        quoted, num, word = match.groups()
        if quoted is not None:
            tokens.append(quoted)
        elif num is not None:
            tokens.append(num)
        elif word is not None:
            tokens.append(word)
    return tokens


def load_powerworld_aux(file_path):
    """
    Parses a PowerWorld Auxiliary (.aux) file and loads grid data structure.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"AUX file not found: {file_path}")

    with open(file_path, 'r') as f:
        content = f.read()

    grid_data = {'buses': [], 'generators': [], 'loads': [], 'branches': []}

    # Extract BUS block
    bus_match = re.search(r'DATA \(BUS,[^\)]*\)\s*\{([^\}]*)\}', content, re.DOTALL)
    if bus_match:
        lines = bus_match.group(1).strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = _parse_aux_line(line)
            if len(parts) >= 6:
                grid_data['buses'].append({
                    'bus_num': int(parts[0]),
                    'bus_name': parts[1],
                    'base_kv': float(parts[2]),
                    'bus_type': 1 if "Slack" in parts[3] else (2 if "PV" in parts[3] else 3),
                    'v_ang': float(parts[4]),
                    'v_mag': float(parts[5])
                })

    # Extract BRANCH block
    branch_match = re.search(r'DATA \(BRANCH,[^\)]*\)\s*\{([^\}]*)\}', content, re.DOTALL)
    if branch_match:
        lines = branch_match.group(1).strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = _parse_aux_line(line)
            if len(parts) >= 7:
                grid_data['branches'].append({
                    'from_bus': int(parts[0]),
                    'to_bus': int(parts[1]),
                    'circuit_id': parts[2],
                    'r': float(parts[3]),
                    'x': float(parts[4]),
                    'b': float(parts[5]),
                    'status': parts[6],
                    'rate_a': float(parts[7]) if len(parts) > 7 else 250.0
                })

    return grid_data


def compute_admittance_matrix(grid_data, num_buses=35):
    """
    Computes the 35x35 complex admittance matrix Y_bus and its magnitude matrix
    (Admittance-Adjacency Matrix, Channel 1).

    Returns:
        Y_bus (np.ndarray): Complex 35x35 matrix
        Y_adj (np.ndarray): Real 35x35 magnitude matrix |Y_bus|
    """
    Y_bus = np.zeros((num_buses, num_buses), dtype=complex)

    for br in grid_data['branches']:
        if br.get('status', 'Closed') != 'Closed':
            continue
        
        i = br['from_bus'] - 1  # 0-indexed
        j = br['to_bus'] - 1

        r = br['r']
        x = br['x']
        b_shunt = br['b']
        tap = br.get('tap', 1.0)

        # Series admittance y = 1 / (r + j*x)
        z = complex(r, x)
        y = 1.0 / z
        b_half = complex(0, b_shunt / 2.0)

        # Off-diagonal elements
        Y_bus[i, j] -= y / tap
        Y_bus[j, i] -= y / tap

        # Diagonal elements
        Y_bus[i, i] += (y / (tap ** 2)) + b_half
        Y_bus[j, j] += y + b_half

    Y_adj = np.abs(Y_bus)
    return Y_bus, Y_adj


if __name__ == '__main__':
    # Unit test / execution check
    grid = generate_35bus_grid(seed=42)
    aux_path = os.path.join('data', 'grid_35bus.aux')
    export_powerworld_aux(grid, aux_path)
    
    parsed_grid = load_powerworld_aux(aux_path)
    Y_bus, Y_adj = compute_admittance_matrix(parsed_grid)

    print(f"Generated Grid: {len(grid['buses'])} buses, {len(grid['branches'])} branches.")
    print(f"Admittance Matrix Shape: {Y_adj.shape}")
    print(f"Y_adj Max Admittance: {Y_adj.max():.4f}, Min: {Y_adj.min():.4f}")
