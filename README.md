# N-1 Contingency Recommendation System for Power Systems
### 3-Channel 2D Convolutional Neural Network (TensorFlow / Keras)

This project develops an end-to-end Machine Learning recommendation system for power grid N-1 contingency analysis using a **3-Channel 2D Convolutional Neural Network (CNN)** built in **TensorFlow / Keras**.

The system models a **35-bus power grid**, exports/imports industry-standard PowerWorld `.aux` files, generates 5,000 synthetic power system samples, trains a Residual 2D CNN model, and recommends which buses or branches require N-1 contingency stress-testing.

---

## ⚡ System Architecture & Input Channel Specifications

The model processes $35 \times 35$ grid matrices representing pairwise interactions and bus injections across 3 distinct physical channels:

- **Input Tensor Shape**: `(Batch, 35, 35, 3)`
  - **Channel 1 (Admittance |Y_bus|)**: Magnitude of the complex bus admittance matrix $|Y_{bus}|$ ($35 \times 35$), capturing grid network topology and structural branch impedances.
  - **Channel 2 (Real Power P)**: Pairwise real power flow $P_{ij}$ across transmission lines and net active power bus injections $P_{ii}$ ($35 \times 35$).
  - **Channel 3 (Reactive Power Q)**: Pairwise reactive power flow $Q_{ij}$ across transmission lines and net reactive power bus injections $Q_{ii}$ ($35 \times 35$).
- **Output Target Matrix Shape**: `(Batch, 35, 35)`
  - Binary label matrix $Y \in \{0, 1\}^{35 \times 35}$ where $1$s indicate buses/branches recommended for N-1 contingency stress-testing and $0$s indicate non-critical elements.

---

## 📊 Dataset & Data Split

- **Total Samples**: 5,000 synthetic operating states simulating load/generation variations.
- **Dataset Split**:
  - **Training Set (60%)**: 3,000 samples (`x_train`, `y_train`)
  - **Validation Set (20%)**: 1,000 samples (`x_val`, `y_val`)
  - **Test Set (20%)**: 1,000 samples (`x_test`, `y_test`)
- **Imbalance / Density**: Positive label density is ~2.25% ($1$s indicating critical contingencies).

---

## 🏆 Model Evaluation Results (1,000 Unseen Test Samples)

The model was evaluated on **1,000 test set samples** (1,225,000 matrix elements):

| Metric | Score | Interpretation |
|---|---|---|
| **Test Accuracy** | **`99.56%`** | Overall matrix element classification accuracy |
| **Test Recall** | **`100.00%`** | **0 missed critical contingencies** (27,640 / 27,640 detected) |
| **Test Precision** | **`83.76%`** | High positive prediction precision |
| **Test F1-Score** | **`0.9116`** | Harmonic mean of Precision & Recall |
| **ROC-AUC Score** | **`0.9993`** | Near-perfect area under ROC curve |

### Confusion Matrix Breakdown:
- **True Positives (TP)**: `27,640` (correctly identified critical contingencies)
- **False Positives (FP)**: `5,360` (extra safety margin recommendations)
- **True Negatives (TN)**: `1,192,000` (correctly unflagged non-critical elements)
- **False Negatives (FN)**: **`0`** (**zero missed blackout risks**)

---

## 📈 Explanation of Performance Curves & Visual Plots

### 1. **Binary Prediction Matrix ($p \ge 0.5$)**
- **What it shows**: Converts the continuous prediction probabilities into a $35 \times 35$ grid of $0$s and $1$s using a $0.5$ probability threshold.
- **Power System Meaning**:
  - **Entry $(i, j) = 1$ (Flagged)**: Recommends **stress-testing transmission line $(i, j)$** under N-1 contingency analysis (or monitoring bus $i$/$j$ on the diagonal).
  - **Entry $(i, j) = 0$ (Unflagged)**: Predicts line $(i, j)$ will **not experience severe overloads or voltage violations** under single-element outages.
- **Why it matters**: Acts as an automated **screening filter**. Instead of executing computationally expensive full-AC power flow simulations on all grid elements, power system operators can focus exclusively on flagged entries.

### 2. **ROC Curve (Receiver Operating Characteristic)**
- **What it shows**: Plots **True Positive Rate (Recall)** against **False Positive Rate** as the probability decision threshold varies from $1.0$ down to $0.0$.
- **Model Score**: **AUC = 0.9993** (Area Under Curve is near 1.0).
- **Power System Meaning**:
  - Measures the CNN's overall ability to discriminate between **critical N-1 contingency risks** and **safe grid operating states**.
  - An AUC of **0.9993** means that if you randomly pick one true critical line outage and one safe line, the model assigns a higher stress risk score to the critical line **99.93% of the time**.

### 3. **Precision-Recall (PR) Curve**
- **What it shows**: Plots **Precision** (y-axis) against **Recall** (x-axis) across different decision thresholds.
- **Why PR is vital for power grid analysis**:
  - Power grid contingency matrices are **heavily imbalanced** (~97.75% $0$s vs ~2.25% $1$s). Standard Accuracy can be misleading (e.g., a dummy model predicting all $0$s gets 97.75% accuracy but misses every blackout risk!).
  - **Recall ($\mathbf{100.00\%}$)**: Answers *"Out of all actual dangerous line outages, how many did the model catch?"* Here, **Recall = 100% (0 missed critical outages)**. In power system reliability, **missing a critical contingency can trigger cascading blackouts**, making 100% Recall the single most critical safety requirement.
  - **Precision ($\mathbf{83.76\%}$)**: Answers *"When the CNN flags a line for stress-testing, how often is it a true critical risk?"* An 83.76% Precision means over 5 out of 6 recommendations are true critical risks, eliminating redundant simulation workload for power engineers.

---

## 📁 File Structure

```
Project1/
├── AGENTS.md               # User project rules & requirements
├── PROJECT_LOG.md          # Phased progress & execution log
├── README.md               # Project documentation (this file)
├── requirements.txt        # Python package dependencies
├── grid_generator.py       # 35-Bus Grid generator & PowerWorld AUX exporter/parser
├── data_generator.py       # 5,000-sample 3-channel dataset generator & N-1 contingency solver
├── inspect_dataset.py      # NPZ Dataset inspection utility
├── model.py                # TensorFlow / Keras 2D CNN architecture & data loader
├── train.py                # Weighted BCE training engine with metrics & checkpointing
├── evaluate.py             # Test set evaluation & classification metrics engine
├── visualize.py            # Plot generator for heatmaps, training curves, & ROC/PR curves
├── contingency_notebook.py # Interactive Marimo Data Science notebook
├── main.py                 # Master CLI orchestrator
├── data/
│   ├── grid_35bus.aux      # Exported PowerWorld AUX file
│   ├── dataset.npz         # 5,000-sample 3-channel dataset (60% Train, 20% Val, 20% Test)
│   ├── training_history.npz# Loss & metric history over 25 epochs
│   └── test_evaluation.json# Quantitative test evaluation report
├── checkpoints/
│   └── best_model.keras    # Saved Keras model checkpoint
└── plots/                  # Generated plots
    ├── 01_input_channels_heatmap.png
    ├── 02_ground_truth_vs_predicted_heatmap.png
    ├── 03_training_curves.png
    └── 04_roc_pr_curves.png
```

---

## 🚀 Getting Started & Execution Guide

### 1. **Run Master Pipeline CLI (`main.py`)**
```bash
# Execute all phases sequentially (Grid Gen -> Data Gen -> Model -> Train -> Evaluate -> Visualize)
python main.py --phase all

# Or run specific individual phases:
python main.py --phase 1    # Phase 1: Grid Model & AUX Exporter
python main.py --phase 2    # Phase 2: Synthetic Data Generator (5,000 samples)
python main.py --phase 3    # Phase 3: Keras 2D CNN Architecture
python main.py --phase 4    # Phase 4: Model Training (25 epochs)
python main.py --phase 5    # Phase 5: Test Evaluation & Plot Generation
```

### 2. **Run Interactive Marimo Notebook (`contingency_notebook.py`)**
```bash
# Launch interactive Marimo notebook editor in browser:
marimo edit contingency_notebook.py

# Or launch as a read-only web application dashboard:
marimo run contingency_notebook.py
```

### 3. **Run Individual Scripts**
```bash
python grid_generator.py    # Generates 35-bus grid model & AUX file
python data_generator.py    # Generates 5,000-sample 3-channel dataset
python inspect_dataset.py   # Inspects NPZ dataset arrays and slices
python model.py             # Verifies Keras model summary & forward pass
python train.py             # Trains model for 25 epochs
python evaluate.py          # Evaluates test set performance
python visualize.py         # Generates plots in plots/ directory
```
