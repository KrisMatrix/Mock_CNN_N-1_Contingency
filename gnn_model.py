"""
gnn_model.py
TensorFlow / Keras Graph Neural Network (GNN) architecture and Graph Data Feature Extractor
for predicting N-1 contingency stress-testing recommendation matrices (35x35)
from 3-channel input matrices (Admittance, Real Power P, Reactive Power Q).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="CustomGNN")
def extract_graph_features(x_matrices):
    """
    Extracts node features, normalized adjacency matrix, and edge features
    from 3-channel input matrix tensor of shape (Batch, 35, 35, 3).

    Channels:
        Channel 0: Admittance magnitude matrix |Y_bus|
        Channel 1: Real power matrix P (p.u.)
        Channel 2: Reactive power matrix Q (p.u.)

    Returns:
        nodes: (Batch, 35, 6) -> Node feature tensor
        adj_norm: (Batch, 35, 35) -> Normalized adjacency matrix A_tilde
        edges: (Batch, 35, 35, 4) -> Edge feature tensor (|Y|, P, Q, S)
    """
    # x_matrices shape: (B, 35, 35, 3)
    ch_y = x_matrices[:, :, :, 0]  # Admittance
    ch_p = x_matrices[:, :, :, 1]  # Real Power
    ch_q = x_matrices[:, :, :, 2]  # Reactive Power

    # Apparent power S = sqrt(P^2 + Q^2)
    ch_s = tf.sqrt(tf.square(ch_p) + tf.square(ch_q) + 1e-8)

    # 1. Edge Features: Stack (|Y|, P, Q, S) -> Shape (B, 35, 35, 4)
    edges = tf.stack([ch_y, ch_p, ch_q, ch_s], axis=-1)

    # 2. Node Features: (B, 35, 6)
    # Diagonals store net power injections P_ii, Q_ii
    p_diag = tf.linalg.diag_part(ch_p)  # (B, 35)
    q_diag = tf.linalg.diag_part(ch_q)  # (B, 35)

    # Row summaries: total admittance sum, total outgoing real & reactive power, node degree
    y_sum = tf.reduce_sum(ch_y, axis=-1)  # (B, 35)
    p_sum = tf.reduce_sum(tf.abs(ch_p), axis=-1)  # (B, 35)
    q_sum = tf.reduce_sum(tf.abs(ch_q), axis=-1)  # (B, 35)
    
    # Binary degree (connected lines count)
    adj_binary = tf.cast(ch_y > 1e-4, dtype=tf.float32)
    degree = tf.reduce_sum(adj_binary, axis=-1)  # (B, 35)

    # Expand dims for stacking: (B, 35, 1)
    nodes = tf.stack([
        p_diag,
        q_diag,
        y_sum,
        p_sum,
        q_sum,
        degree
    ], axis=-1)

    # 3. Normalized Adjacency Matrix A_tilde = D^{-1/2} (A + I) D^{-1/2}
    batch_size = tf.shape(x_matrices)[0]
    num_nodes = 35
    eye = tf.eye(num_nodes, batch_shape=[batch_size])  # (B, 35, 35)

    adj_self = adj_binary + eye  # Add self-loops
    d = tf.reduce_sum(adj_self, axis=-1)  # Degree vector (B, 35)
    d_inv_sqrt = tf.math.rsqrt(tf.maximum(d, 1e-8))  # D^{-1/2}
    d_mat = tf.linalg.diag(d_inv_sqrt)  # (B, 35, 35)

    # A_tilde = D^{-1/2} * (A + I) * D^{-1/2}
    adj_norm = tf.matmul(tf.matmul(d_mat, adj_self), d_mat)

    return nodes, adj_norm, edges


@keras.utils.register_keras_serializable(package="CustomGNN")
class GraphConvLayer(layers.Layer):
    """
    Spatial/Spectral Graph Convolutional Layer in Keras.
    Message Passing: H^{(l+1)} = LeakyReLU( A_tilde * H^{(l)} * W_neigh + H^{(l)} * W_self + b )
    """
    def __init__(self, units, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units

    def build(self, input_shape):
        # input_shape is list: [node_shape (B, 35, din), adj_shape (B, 35, 35)]
        in_dim = input_shape[0][-1]
        self.w_neigh = self.add_weight(
            shape=(in_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="w_neigh"
        )
        self.w_self = self.add_weight(
            shape=(in_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="w_self"
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
        self.bn = layers.BatchNormalization(name=f"{self.name}_bn")
        self.act = layers.LeakyReLU(alpha=0.2, name=f"{self.name}_act")

    def call(self, inputs):
        nodes, adj_norm = inputs  # nodes: (B, 35, din), adj_norm: (B, 35, 35)

        # Neighbor aggregation: A_tilde * nodes -> (B, 35, din)
        aggregated = tf.matmul(adj_norm, nodes)

        # Transformation
        h_neigh = tf.matmul(aggregated, self.w_neigh)  # (B, 35, units)
        h_self = tf.matmul(nodes, self.w_self)        # (B, 35, units)

        out = h_neigh + h_self + self.bias
        out = self.bn(out)
        out = self.act(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


@keras.utils.register_keras_serializable(package="CustomGNN")
class PairPredictionHead(layers.Layer):
    """
    Pairwise Recommendation Prediction Head.
    Constructs pair representations z_ij = [h_i || h_j || (h_i * h_j) || e_ij]
    and maps to a (35, 35) contingency recommendation logit matrix.
    """
    def __init__(self, hidden_dim=64, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        # input_shape: [node_embeddings (B, 35, d), edge_features (B, 35, 35, 4)]
        self.dense1 = layers.Dense(self.hidden_dim, activation=None, name="pair_dense1")
        self.bn1 = layers.BatchNormalization(name="pair_bn1")
        self.act1 = layers.LeakyReLU(alpha=0.2, name="pair_act1")

        self.dense2 = layers.Dense(32, activation=None, name="pair_dense2")
        self.bn2 = layers.BatchNormalization(name="pair_bn2")
        self.act2 = layers.LeakyReLU(alpha=0.2, name="pair_act2")

        self.out_dense = layers.Dense(1, activation=None, name="pair_out")

    def call(self, inputs):
        h, edges = inputs  # h: (B, 35, d), edges: (B, 35, 35, 4)
        num_nodes = 35

        # Broadcast node embeddings for all (i, j) pairs
        # h_i shape: (B, 35, 1, d), h_j shape: (B, 1, 35, d)
        h_i = tf.expand_dims(h, axis=2)  # (B, 35, 1, d)
        h_j = tf.expand_dims(h, axis=1)  # (B, 1, 35, d)

        # Tile to (B, 35, 35, d)
        h_i_tiled = tf.tile(h_i, [1, 1, num_nodes, 1])
        h_j_tiled = tf.tile(h_j, [1, num_nodes, 1, 1])

        # Hadamard product
        h_prod = h_i_tiled * h_j_tiled

        # Pair feature vector z_ij: concat [h_i, h_j, h_i * h_j, edges]
        pair_features = tf.concat([h_i_tiled, h_j_tiled, h_prod, edges], axis=-1)  # (B, 35, 35, 3d + 4)

        # Pass through MLP
        x = self.dense1(pair_features)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.dense2(x)
        x = self.bn2(x)
        x = self.act2(x)

        logits_4d = self.out_dense(x)  # (B, 35, 35, 1)
        logits_3d = tf.squeeze(logits_4d, axis=-1, name="contingency_logits")  # (B, 35, 35)

        return logits_3d

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


def create_gnn_model(input_shape=(35, 35, 3)):
    """
    Builds a TensorFlow/Keras Graph Neural Network (GNN) model for N-1 Contingency Recommendation.
    Input Shape: (Batch, 35, 35, 3)
    Output Shape: (Batch, 35, 35) -> unnormalized logits
    """
    inputs = keras.Input(shape=input_shape, name="grid_matrices")

    # 1. Feature Extraction Layer (Lambda / Functional extraction)
    nodes, adj_norm, edges = layers.Lambda(
        extract_graph_features,
        name="graph_feature_extractor"
    )(inputs)

    # 2. Graph Convolution Layer 1 (6 node features -> 32 dimensions)
    h1 = GraphConvLayer(32, name="gcn_block1")([nodes, adj_norm])

    # 3. Graph Convolution Layer 2 with Residual Connection (32 -> 64 dimensions)
    h2 = GraphConvLayer(64, name="gcn_block2")([h1, adj_norm])

    # 4. Graph Convolution Layer 3 with Residual Connection (64 -> 128 dimensions)
    h3 = GraphConvLayer(128, name="gcn_block3")([h2, adj_norm])

    # 5. Bottleneck Graph Convolution Layer (128 -> 64 dimensions)
    h4 = GraphConvLayer(64, name="gcn_bottleneck")([h3, adj_norm])

    # 6. Pairwise Prediction Head (Node embeddings + Edge features -> 35x35 Logits Matrix)
    outputs = PairPredictionHead(hidden_dim=64, name="pair_prediction_head")([h4, edges])

    model = keras.Model(inputs=inputs, outputs=outputs, name="ContingencyGNN")
    return model


if __name__ == '__main__':
    # Verify TensorFlow Keras GNN Model creation and forward pass
    print("Initializing ContingencyGNN Architecture...")
    model = create_gnn_model(input_shape=(35, 35, 3))
    model.summary()

    # Dummy batch of 4 samples (4, 35, 35, 3)
    dummy_input = tf.random.normal((4, 35, 35, 3))
    dummy_output = model(dummy_input)

    print(f"\nGNN Forward Pass Verification:")
    print(f"  Input Tensor Shape:  {dummy_input.shape}")
    print(f"  Output Logits Shape: {dummy_output.shape}")
    assert dummy_output.shape == (4, 35, 35), "Output shape mismatch!"
    print("TensorFlow / Keras GNN Forward pass verified successfully!")
