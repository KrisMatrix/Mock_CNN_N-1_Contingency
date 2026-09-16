# Project Execution & Activity Log

## Project Goal
Develop a 3-channel Convolutional Neural Network (CNN) recommendation system for N-1 contingency stress testing in a 35-bus power system, adhering to specs in `AGENTS.md`.

---

## Phased Project Roadmap & Status

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Virtual Environment Setup, 35-Bus Grid Generator, PowerWorld AUX Exporter & Parser, Admittance Matrix ($Y_{bus}$) Computation | ✅ Completed |
| **Phase 2** | Synthetic Data Generator (5,000 samples, 3 channels: Admittance, Real Power $P$, Reactive Power $Q$, N-1 labels) & Dataset Split | ✅ Completed |
| **Phase 3** | TensorFlow / Keras 2D CNN Architecture Design (`(35, 35, 3)` $\rightarrow$ `(35, 35)`) | ✅ Completed |
| **Phase 4** | Model Training & Validation Engine (TensorFlow Keras) with Checkpointing | ✅ Completed |
| **Phase 5** | Model Evaluation (Test set) & Comprehensive Visualization Suite | ✅ Completed |
| **Phase 6** | End-to-End Pipeline Integration (`main.py`) & Marimo Exploratory Notebook (`contingency_notebook.py`) | ✅ Completed |

---

## Activity Log

### [2026-09-15] Documentation & Interactive Notebook Finalized
- Created [`README.md`](file:///c:/Users/kkGamingPC/Documents/Project1/README.md) containing project architecture, input channel specifications, test performance results table, domain explanations of ROC, PR, and Binary Prediction curves, file structure layout, and execution instructions.
- Finalized interactive Marimo notebook [`contingency_notebook.py`](file:///c:/Users/kkGamingPC/Documents/Project1/contingency_notebook.py) for exploratory data science analysis.



### [2026-09-15] Phase 6 Completed (Entire Pipeline Verified End-to-End)
- Built `main.py` CLI orchestrator supporting execution of individual phases (`--phase 1` .. `--phase 5`) or end-to-end execution (`--phase all`).
- Verified all requirements from `AGENTS.md`:
  - 35-bus electric grid model & PowerWorld `.aux` format generator (`grid_generator.py`)
  - 5,000 synthetic samples with 3 matrix channels: Admittance $|Y_{bus}|$, Real Power $P$, Reactive Power $Q$ (`data_generator.py`)
  - 60% Train / 20% Val / 20% Test split (`data/dataset.npz`)
  - TensorFlow / Keras 2D CNN recommendation model architecture (`model.py`)
  - Training loop with weighted BCE loss & checkpointing (`train.py`)
  - Test set evaluation engine (`evaluate.py`)
  - High-resolution visualization suite generating plots in `plots/` (`visualize.py`).


### [2026-09-15] Phase 5 Completed
- Created `evaluate.py` and evaluated model on **1,000 un-seen Test set samples**:
  - **Test Loss**: `0.0213`
  - **Accuracy**: `99.56%`
  - **Precision**: `83.76%`
  - **Recall**: `100.00%` (0 missed critical N-1 contingencies out of 27,640 targets!)
  - **F1-Score**: `0.9116`
  - **ROC-AUC Score**: `0.9993`
  - Saved evaluation output to `data/test_evaluation.json`.
- Created `visualize.py` and generated 4 high-resolution plots in `plots/`:
  1. `plots/01_input_channels_heatmap.png`
  2. `plots/02_ground_truth_vs_predicted_heatmap.png`
  3. `plots/03_training_curves.png`
  4. `plots/04_roc_pr_curves.png`


### [2026-09-15] Phase 4 Completed
- Created and executed `train.py` using **TensorFlow / Keras**:
  - Implemented weighted binary cross-entropy loss function (`pos_weight=15.0`) to compensate for positive label matrix sparsity.
  - Configured `AdamW` optimizer, `ReduceLROnPlateau`, `EarlyStopping`, and custom `F1ScoreMetric`.
  - Trained model for 25 epochs:
    - **Val Loss**: dropped from `0.1709` $\rightarrow$ `0.0796`
    - **Val Accuracy**: `98.79%`
    - **Val Recall**: `93.50%`
    - **Val AUC**: `0.9968`
    - **Val F1-Score**: `0.7501`
  - Saved model checkpoint to `checkpoints/best_model.keras` and history to `data/training_history.npz`.


### [2026-09-15] Refactored to TensorFlow / Keras (Phase 3 Completed)
- Rewrote `model.py` using **TensorFlow / Keras** as requested:
  - Model architecture: `ContingencyCNN` with Residual 2D Convolutional layers (`Conv2D`, `BatchNormalization`, `LeakyReLU`, `Add`), bottleneck projection, and `(35, 35)` output reshape layer.
  - 363,425 trainable parameters.
  - Input format: `(Batch, 35, 35, 3)` (`channels_last`).
  - Implemented `load_dataset()` to automatically transpose input matrices for TensorFlow (`(N, 3, 35, 35)` $\rightarrow$ `(N, 35, 35, 3)`).
  - Verified model summary and forward pass (`(4, 35, 35, 3)` $\rightarrow$ `(4, 35, 35)`).



### [2026-09-15] Phase 2 Completed
- Created and executed `data_generator.py`:
  - Generated **5,000 samples** with 3 matrix channels `(3, 35, 35)` (Channel 1: Admittance magnitude, Channel 2: Real power $P_{ij}$, Channel 3: Reactive power $Q_{ij}$).
  - Evaluated N-1 contingency stress severity indices across all line outage scenarios to produce ground-truth $35 \times 35$ binary target label matrices.
  - Formatted split: **60% Train (3,000)**, **20% Val (1,000)**, **20% Test (1,000)**.
  - Compressed and saved dataset to `data/dataset.npz` (positive label ratio ~2.25%).

### [2026-09-15] Phase 1 Completed
- Created `PROJECT_LOG.md` to maintain active record of plans, tasks, and phase completion status.
- User created clean Python virtual environment (`env`), installed all required dependencies (`torch`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `pandas`), and generated `requirements.txt`.
- Implemented and verified `grid_generator.py`:
  - `generate_35bus_grid()`: 35-bus power system topology generator (35 buses, 6 generators, 25 loads, 55 transmission line/transformer branches).
  - `export_powerworld_aux()`: Successfully generated `data/grid_35bus.aux` in PowerWorld AUX format (`BUS`, `GEN`, `LOAD`, `BRANCH` data blocks).
  - `load_powerworld_aux()`: Verified parser for AUX data files with robust pattern tokenization.
  - `compute_admittance_matrix()`: Calculated $35 \times 35$ complex admittance matrix $Y_{bus}$ and magnitude matrix $|Y_{bus}|$ (Channel 1 Admittance-Adjacency matrix).




