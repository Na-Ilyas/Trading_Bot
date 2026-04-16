"""
Hybrid GCN-BiLSTM Model
════════════════════════
Manual GCN implementation (no Spektral version issues).
Sequential GCN layers, direction prediction (classification).
Optional improvements: Temporal Attention, Residual Connections,
Label Smoothing, Learnable Adjacency.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Bidirectional, LSTM, Layer,
    GlobalAveragePooling1D, Concatenate, BatchNormalization, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import config as C


class GCNLayer(Layer):
    """Graph Convolutional layer: H' = sigma(A_hat . H . W + b)"""
    def __init__(self, units, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.act = tf.keras.activations.get(activation)

    def build(self, input_shape):
        feat_dim = input_shape[0][-1]
        self.W = self.add_weight(shape=(feat_dim, self.units),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros", trainable=True)

    def call(self, inputs):
        X, A = inputs
        support = tf.matmul(X, self.W)
        output = tf.matmul(A, support)
        return self.act(output + self.b)


class TemporalAttention(Layer):
    """Bahdanau-style attention over time steps."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W_att = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer="glorot_uniform", trainable=True)
        self.b_att = self.add_weight(shape=(self.units,),
                                     initializer="zeros", trainable=True)
        self.v = self.add_weight(shape=(self.units, 1),
                                 initializer="glorot_uniform", trainable=True)

    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        score = tf.nn.tanh(tf.matmul(inputs, self.W_att) + self.b_att)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.v), axis=1)
        context = tf.reduce_sum(inputs * attention_weights, axis=1)
        return context


class LearnableAdjacency(Layer):
    """Learns a soft adjacency matrix from trainable node embeddings."""
    def __init__(self, window_size, embed_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.node_emb = self.add_weight(
            shape=(self.window_size, self.embed_dim),
            initializer="glorot_uniform", trainable=True, name="node_emb"
        )
        self.alpha = self.add_weight(
            shape=(1,), initializer=tf.keras.initializers.Constant(0.5),
            trainable=True, name="alpha"
        )

    def call(self, inputs):
        # inputs: fixed adjacency (batch, W, W)
        A_fixed = inputs
        sim = tf.matmul(self.node_emb, self.node_emb, transpose_b=True)
        A_learned = tf.nn.softmax(sim, axis=-1)
        alpha = tf.nn.sigmoid(self.alpha)
        A_combined = alpha * A_fixed + (1.0 - alpha) * A_learned
        return A_combined


def build_temporal_adjacency(window_size: int) -> np.ndarray:
    A = np.zeros((window_size, window_size), dtype=np.float32)
    for i in range(window_size - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    A += np.eye(window_size, dtype=np.float32)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-10))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


def create_hybrid_model(window_size: int, n_features: int) -> Model:
    tf.random.set_seed(C.SYNTHETIC_SEED)
    x_input = Input(shape=(window_size, n_features), name="features")
    a_input = Input(shape=(window_size, window_size), name="adjacency")

    # Optionally learn adjacency
    if C.USE_LEARNABLE_ADJ:
        adj = LearnableAdjacency(window_size, name="learn_adj")(a_input)
    else:
        adj = a_input

    # GCN Branch (SEQUENTIAL)
    gcn_1 = GCNLayer(C.GCN_UNITS_1, activation="relu", name="gcn_1")([x_input, adj])
    gcn_1 = Dropout(C.DROPOUT_RATE)(gcn_1)
    gcn_2 = GCNLayer(C.GCN_UNITS_2, activation="relu", name="gcn_2")([gcn_1, adj])
    gcn_pool = GlobalAveragePooling1D()(gcn_2)
    gcn_pool = Dropout(C.DROPOUT_RATE)(gcn_pool)

    # BiLSTM Branch
    lstm_1 = LSTM(C.LSTM_UNITS_1, return_sequences=True)(x_input)
    lstm_1 = BatchNormalization()(lstm_1)
    lstm_2 = LSTM(C.LSTM_UNITS_2, return_sequences=True)(lstm_1)
    lstm_2 = BatchNormalization()(lstm_2)

    if C.USE_ATTENTION:
        bi_lstm = Bidirectional(LSTM(C.BILSTM_UNITS, return_sequences=True))(lstm_2)
        bi_lstm = Dropout(C.DROPOUT_RATE)(bi_lstm)
        bi_lstm = TemporalAttention(C.ATTENTION_UNITS, name="temp_attn")(bi_lstm)
    else:
        bi_lstm = Bidirectional(LSTM(C.BILSTM_UNITS))(lstm_2)
        bi_lstm = Dropout(C.DROPOUT_RATE)(bi_lstm)

    # Fusion
    combined = Concatenate()([gcn_pool, bi_lstm])
    x = Dense(64, activation="relu")(combined)
    x = Dropout(C.DROPOUT_RATE)(x)

    if C.USE_RESIDUAL:
        residual = x
        x = Dense(64, activation="relu")(x)
        x = Dropout(C.DROPOUT_RATE)(x)
        x = Add()([x, residual])

    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="sigmoid", name="direction")(x)

    model = Model(inputs=[x_input, a_input], outputs=output)

    # Loss with optional label smoothing
    if C.LABEL_SMOOTHING > 0:
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=C.LABEL_SMOOTHING)
    else:
        loss = "binary_crossentropy"

    model.compile(optimizer=Adam(learning_rate=C.LEARNING_RATE),
                  loss=loss, metrics=["accuracy"])
    return model


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PAT,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1)
    ]
