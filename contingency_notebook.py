import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import os
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
        confusion_matrix
    )
    import marimo as mo

    from grid_generator import (
        generate_35bus_grid,
        export_powerworld_aux,
        load_powerworld_aux,
        compute_admittance_matrix
    )
    from model import create_cnn_model, load_dataset
    from train import weighted_bce_loss, F1ScoreMetric

    return (
        F1ScoreMetric,
        accuracy_score,
        compute_admittance_matrix,
        confusion_matrix,
        create_cnn_model,
        export_powerworld_aux,
        f1_score,
        generate_35bus_grid,
        keras,
        load_dataset,
        load_powerworld_aux,
        mo,
        np,
        os,
        pd,
        plt,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        weighted_bce_loss,
    )


@app.cell
def _(mo):
    mo.md("""
    # ⚡ Exploratory Notebook: N-1 Contingency Recommendation CNN
    ### Power System Modeling, 3-Channel Data Analysis, Keras Training & Evaluation

    This interactive Data Science notebook explores:
    1. PowerWorld AUX grid parsing and complex admittance matrix $Y_{bus}$ calculation.
    2. 3-Channel matrix data inspection ($Y_{bus}$, Real Power $P$, Reactive Power $Q$) and ground truth target matrix $Y$.
    3. Model building, training, and validation live inside the notebook.
    4. Dynamic model evaluation and performance visualization.

    ---
    """)
    return


@app.cell
def _(
    compute_admittance_matrix,
    export_powerworld_aux,
    generate_35bus_grid,
    load_powerworld_aux,
    mo,
    os,
    pd,
):
    # 1. Generate 35-bus grid model & export PowerWorld AUX
    raw_grid = generate_35bus_grid(seed=42)
    aux_path = os.path.join('data', 'grid_35bus.aux')
    export_powerworld_aux(raw_grid, aux_path)

    # 2. Parse PowerWorld AUX file using load_powerworld_aux()
    parsed_grid = load_powerworld_aux(aux_path)

    # 3. Convert parsed bus and branch attributes to pandas DataFrames for inspection
    df_buses = pd.DataFrame(parsed_grid['buses'])
    df_branches = pd.DataFrame(parsed_grid['branches'])

    # 4. Compute complex admittance matrix Y_bus and magnitude matrix Y_adj
    Y_bus, Y_adj = compute_admittance_matrix(parsed_grid)

    mo.md(
        f"""
        ## 1️⃣ Phase 1: PowerWorld AUX Grid Parsing (`load_powerworld_aux()`)
    
        The function `load_powerworld_aux('{aux_path}')` reads standard industry PowerWorld `.aux` text files, extracting `BUS`, `GEN`, `LOAD`, and `BRANCH` data blocks.

        ### Parsed Bus Data Sample (First 8 Buses):
        {df_buses.head(8).to_markdown(index=False)}

        ---

        ### Parsed Transmission Branch Data Sample (First 8 Lines/Transformers):
        {df_branches.head(8).to_markdown(index=False)}

        ---

        ### Admittance Matrix ($Y_{{bus}}$) Computation:
        - **$Y_{{bus}}$ Complex Matrix Shape**: `{Y_bus.shape}`
        - **Max Admittance $|Y_{{bus}}|$**: `{Y_adj.max():.4f}` per-unit
        """
    )
    return


@app.cell
def _(load_dataset, mo, plt):
    # Load dataset
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset('data/dataset.npz')

    # Inspect sample 0
    sample_0_x = x_train[0]  # Shape (35, 35, 3)
    sample_0_y = y_train[0]  # Shape (35, 35)

    # Plot Sample 0 Channels and Target Label
    fig_data, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    im0 = axes[0].imshow(sample_0_x[:, :, 0], cmap='viridis')
    axes[0].set_title("Ch 1: Admittance |Y_bus|", fontsize=11, fontweight='bold')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sample_0_x[:, :, 1], cmap='plasma')
    axes[1].set_title("Ch 2: Real Power P", fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(sample_0_x[:, :, 2], cmap='coolwarm')
    axes[2].set_title("Ch 3: Reactive Power Q", fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(sample_0_y, cmap='Blues', vmin=0, vmax=1)
    axes[3].set_title("Target Label Matrix Y", fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=axes[3])

    plt.tight_layout()

    pos_ratio = (y_train.sum() / y_train.size) * 100

    mo.md(
        f"""
        ## 2️⃣ Phase 2: Exploratory Data Analysis (`X_train` & `y_train`)

        ### Dataset Tensor Dimensions:
        - **`X_train` Shape**: `{x_train.shape}` — 3,000 samples, $35 \\times 35$ grid size, **3 matrix channels**.
        - **`y_train` Shape**: `{y_train.shape}` — 3,000 binary target matrices ($35 \\times 35$).
        - **`X_val` / `X_test` Shapes**: `{x_val.shape}` / `{x_test.shape}` (1,000 validation & 1,000 test samples).
        - **Class Distribution**: **{pos_ratio:.2f}%** positive entries ($1$s indicating critical N-1 contingency recommendations).

        ---

        ### What the Viewer is Seeing in Sample 0:
        1. **Channel 1 (Admittance |Y_bus|)**: Structural grid impedance/connectivity magnitude between bus $i$ and bus $j$.
        2. **Channel 2 (Real Power P)**: Pairwise active power flow $P_{{ij}}$ across branches and net bus injections $P_{{ii}}$.
        3. **Channel 3 (Reactive Power Q)**: Pairwise reactive power flow $Q_{{ij}}$ across branches and net bus injections $Q_{{ii}}$.
        4. **Target Label Matrix Y**: Ground truth matrix where $1$s flag critical buses/branches requiring stress testing.

        {mo.as_html(fig_data)}
        """
    )
    return x_test, x_train, x_val, y_test, y_train, y_val


@app.cell
def _(
    F1ScoreMetric,
    create_cnn_model,
    keras,
    mo,
    plt,
    weighted_bce_loss,
    x_train,
    x_val,
    y_train,
    y_val,
):
    # Build Keras 2D CNN Model
    model = create_cnn_model(input_shape=(35, 35, 3))

    # Compile with weighted loss & metrics
    loss_fn = weighted_bce_loss(pos_weight=15.0)
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.0),
            keras.metrics.Precision(name='precision', thresholds=0.0),
            keras.metrics.Recall(name='recall', thresholds=0.0),
            keras.metrics.AUC(name='auc', from_logits=True),
            F1ScoreMetric(name='f1_score', threshold=0.0)
        ]
    )

    # Execute training live in notebook for 10 epochs
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=0
    )

    # Save trained checkpoint
    model.save('checkpoints/best_model.keras')

    # Plot Training Curves directly in cell
    epochs = range(1, len(history.history['loss']) + 1)
    fig_hist, ax_hist = plt.subplots(1, 2, figsize=(14, 4.5))

    ax_hist[0].plot(epochs, history.history['loss'], 'b-o', label='Train Loss')
    ax_hist[0].plot(epochs, history.history['val_loss'], 'r--s', label='Val Loss')
    ax_hist[0].set_title("Weighted Loss over Epochs", fontsize=11, fontweight='bold')
    ax_hist[0].set_xlabel("Epoch")
    ax_hist[0].set_ylabel("Loss")
    ax_hist[0].legend()
    ax_hist[0].grid(True, linestyle='--', alpha=0.5)

    ax_hist[1].plot(epochs, history.history['f1_score'], 'b-o', label='Train F1')
    ax_hist[1].plot(epochs, history.history['val_f1_score'], 'r--s', label='Val F1')
    ax_hist[1].set_title("F1-Score over Epochs", fontsize=11, fontweight='bold')
    ax_hist[1].set_xlabel("Epoch")
    ax_hist[1].set_ylabel("F1 Score")
    ax_hist[1].legend()
    ax_hist[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    final_loss = history.history['val_loss'][-1]
    final_f1 = history.history['val_f1_score'][-1]

    mo.md(
        f"""
        ## 3️⃣ Phase 3 & 4: Live Model Building & Training (10 Epochs)

        The `ContingencyCNN` model was compiled and trained live inside this cell on **3,000 training samples** and evaluated on **1,000 validation samples**.

        ### Training Summary:
        - **Final Validation Loss**: `{final_loss:.4f}`
        - **Final Validation F1-Score**: `{final_f1:.4f}`

        {mo.as_html(fig_hist)}
        """
    )
    return (model,)


@app.cell
def _(
    accuracy_score,
    confusion_matrix,
    f1_score,
    model,
    np,
    precision_score,
    recall_score,
    roc_auc_score,
    x_test,
    y_test,
):
    # Run live model inference on 1,000 test set samples
    logits_test = model.predict(x_test, batch_size=32, verbose=0)
    probs_test = 1.0 / (1.0 + np.exp(-logits_test))

    y_true_flat = y_test.flatten()
    y_prob_flat = probs_test.flatten()
    y_pred_flat = (y_prob_flat >= 0.5).astype(np.float32)

    # Compute live test metrics
    acc = accuracy_score(y_true_flat, y_pred_flat)
    prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    auc_val = roc_auc_score(y_true_flat, y_prob_flat)

    cm = confusion_matrix(y_true_flat, y_pred_flat)
    tn, fp, fn, tp = cm.ravel()
    return (
        acc,
        auc_val,
        f1,
        fn,
        fp,
        prec,
        probs_test,
        rec,
        tn,
        tp,
        y_prob_flat,
        y_true_flat,
    )


@app.cell
def _(acc, auc_val, f1, fn, fp, mo, prec, rec, tn, tp):
    mo.md(f"""
    ## 4️⃣ Phase 5: Live Test Set Evaluation Results (1,000 Test Samples)

    The trained model was evaluated dynamically on **1,000 unseen test samples** (1,225,000 matrix elements):

    | Classification Metric | Score | Performance Details |
    |---|---|---|
    | **Test Accuracy** | **{acc*100:.2f}%** | Matrix cell prediction accuracy |
    | **Test Recall** | **{rec*100:.2f}%** | **{fn} missed contingencies** ({tp:,} True Positives identified) |
    | **Test Precision** | **{prec*100:.2f}%** | Positive predictive accuracy |
    | **Test F1-Score** | **{f1:.4f}** | Harmonic mean of Precision & Recall |
    | **ROC-AUC Score** | **{auc_val:.4f}** | Area under ROC Curve |

    ---

    ### Confusion Matrix Breakdown:
    - **True Positives (TP)**: `{tp:,}` (correctly flagged N-1 contingencies)
    - **False Positives (FP)**: `{fp:,}` (extra stress-test recommendations)
    - **True Negatives (TN)**: `{tn:,}` (correctly unflagged elements)
    - **False Negatives (FN)**: `{fn:,}` (missed contingencies)
    """)
    return


@app.cell
def _(
    mo,
    plt,
    precision_recall_curve,
    probs_test,
    roc_curve,
    y_prob_flat,
    y_test,
    y_true_flat,
):
    # Plot 1: Prediction Heatmap Comparison
    fig_preds, axes_p = plt.subplots(1, 3, figsize=(16, 5))

    im_gt = axes_p[0].imshow(y_test[0], cmap='Blues', vmin=0, vmax=1)
    axes_p[0].set_title("Ground Truth Target Matrix Y", fontsize=11, fontweight='bold')
    plt.colorbar(im_gt, ax=axes_p[0])

    im_pr = axes_p[1].imshow(probs_test[0], cmap='YlOrRd', vmin=0, vmax=1)
    axes_p[1].set_title("Predicted Probabilities", fontsize=11, fontweight='bold')
    plt.colorbar(im_pr, ax=axes_p[1])

    im_th = axes_p[2].imshow((probs_test[0] >= 0.5).astype(int), cmap='Greens', vmin=0, vmax=1)
    axes_p[2].set_title("Binary Prediction (p >= 0.5)", fontsize=11, fontweight='bold')
    plt.colorbar(im_th, ax=axes_p[2])

    plt.tight_layout()

    # Plot 2: ROC & PR Curves
    fig_roc, axes_roc = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
    axes_roc[0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve')
    axes_roc[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes_roc[0].set_title("ROC Curve", fontsize=11, fontweight='bold')
    axes_roc[0].set_xlabel("False Positive Rate")
    axes_roc[0].set_ylabel("True Positive Rate")
    axes_roc[0].legend()
    axes_roc[0].grid(True, linestyle='--', alpha=0.5)

    prec_pts, rec_pts, _ = precision_recall_curve(y_true_flat, y_prob_flat)
    axes_roc[1].plot(rec_pts, prec_pts, color='green', lw=2, label='PR Curve')
    axes_roc[1].set_title("Precision-Recall Curve", fontsize=11, fontweight='bold')
    axes_roc[1].set_xlabel("Recall")
    axes_roc[1].set_ylabel("Precision")
    axes_roc[1].legend()
    axes_roc[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    mo.md(
        f"""
        ## 5️⃣ Phase 5: Dynamic Prediction Heatmaps & ROC/PR Curves

        ### Sample 0 Contingency Prediction Heatmap Comparison:
        {mo.as_html(fig_preds)}

        ---

        ### ROC & Precision-Recall Performance Curves:
        {mo.as_html(fig_roc)}
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
