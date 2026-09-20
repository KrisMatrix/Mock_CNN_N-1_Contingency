"""
train_gnn.py
Training engine for N-1 Contingency Recommendation Graph Neural Network (GNN) using TensorFlow / Keras.
Features weighted binary cross-entropy loss, AdamW optimizer, learning rate scheduling,
metric tracking (Accuracy, Precision, Recall, AUC, F1), and model checkpointing.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from gnn_model import create_gnn_model, GraphConvLayer, PairPredictionHead
from model import load_dataset
from train import weighted_bce_loss, F1ScoreMetric


def train_gnn_model(epochs=25, batch_size=32, lr=1e-3, pos_weight=15.0, dataset_path='data/dataset.npz', checkpoint_dir='checkpoints'):
    """
    Trains the TensorFlow/Keras ContingencyGNN model on 60% train data and evaluates on 20% val data.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 1. Load Data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset(dataset_path)

    # 2. Instantiate Model
    model = create_gnn_model(input_shape=(35, 35, 3))

    # 3. Compile Model
    optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    loss_function = weighted_bce_loss(pos_weight=pos_weight)

    metrics = [
        keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.0),
        keras.metrics.Precision(name='precision', thresholds=0.0),
        keras.metrics.Recall(name='recall', thresholds=0.0),
        keras.metrics.AUC(name='auc', from_logits=True),
        F1ScoreMetric(name='f1_score', threshold=0.0)
    ]

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    # 4. Define Callbacks
    checkpoint_path = os.path.join(checkpoint_dir, 'best_gnn_model.keras')
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"\nStarting GNN Model Training for {epochs} epochs...")
    print(f"  Train samples: {len(x_train)}, Val samples: {len(x_val)}")
    print(f"  Batch size: {batch_size}, Initial LR: {lr}, Positive Weight: {pos_weight}\n")

    # 5. Train Model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Save Training History for Visualization
    history_path = os.path.join('data', 'gnn_training_history.npz')
    np.savez_compressed(
        history_path,
        loss=np.array(history.history['loss']),
        val_loss=np.array(history.history['val_loss']),
        accuracy=np.array(history.history['accuracy']),
        val_accuracy=np.array(history.history['val_accuracy']),
        precision=np.array(history.history['precision']),
        val_precision=np.array(history.history['val_precision']),
        recall=np.array(history.history['recall']),
        val_recall=np.array(history.history['val_recall']),
        auc=np.array(history.history['auc']),
        val_auc=np.array(history.history['val_auc']),
        f1_score=np.array(history.history['f1_score']),
        val_f1_score=np.array(history.history['val_f1_score'])
    )

    print(f"\nGNN Training completed successfully!")
    print(f"  Best GNN Model Checkpoint saved to: {checkpoint_path}")
    print(f"  GNN Training History saved to:     {history_path}")

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GNN Model for N-1 Contingency Recommendation")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    args = parser.parse_args()

    train_gnn_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
