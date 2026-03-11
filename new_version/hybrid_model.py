"""
Hybrid GCN-BiLSTM Model
════════════════════════
Manual GCN implementation (no Spektral version issues).
Sequential GCN layers, direction prediction (classification).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Bidirectional, LSTM, Layer,
    GlobalAveragePooling1D, Concatenate, BatchNormalization
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


def build_temporal_adjacency(window_size: int) -> np.ndarray:
    A = np.zeros((window_size, window_size), dtype=np.float32)
    for i in range(window_size - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    A += np.eye(window_size, dtype=np.float32)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-10))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


def create_hybrid_model(window_size: int, n_features: int) -> Model:
    x_input = Input(shape=(window_size, n_features), name="features")
    a_input = Input(shape=(window_size, window_size), name="adjacency")

    # GCN Branch (SEQUENTIAL)
    gcn_1 = GCNLayer(C.GCN_UNITS_1, activation="relu", name="gcn_1")([x_input, a_input])
    gcn_1 = Dropout(C.DROPOUT_RATE)(gcn_1)
    gcn_2 = GCNLayer(C.GCN_UNITS_2, activation="relu", name="gcn_2")([gcn_1, a_input])
    gcn_pool = GlobalAveragePooling1D()(gcn_2)
    gcn_pool = Dropout(C.DROPOUT_RATE)(gcn_pool)

    # BiLSTM Branch
    lstm_1 = LSTM(C.LSTM_UNITS_1, return_sequences=True)(x_input)
    lstm_1 = BatchNormalization()(lstm_1)
    lstm_2 = LSTM(C.LSTM_UNITS_2, return_sequences=True)(lstm_1)
    lstm_2 = BatchNormalization()(lstm_2)
    bi_lstm = Bidirectional(LSTM(C.BILSTM_UNITS))(lstm_2)
    bi_lstm = Dropout(C.DROPOUT_RATE)(bi_lstm)

    # Fusion
    combined = Concatenate()([gcn_pool, bi_lstm])
    x = Dense(64, activation="relu")(combined)
    x = Dropout(C.DROPOUT_RATE)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="sigmoid", name="direction")(x)

    model = Model(inputs=[x_input, a_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=C.LEARNING_RATE),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PAT,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1)
    ]
