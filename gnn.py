import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Concatenate
from spektral.layers import GCNConv

def create_hybrid_gnn(n_nodes, n_features):
    """
    Hybrid GNN-BiLSTM architecture as proposed by Das et al. (2026).
    Captures structural dependencies between sentiment and price data.
    """
    # X: Node features (Price, RSI, Sentiment Score)
    # A: Adjacency matrix (Relationship between features)
    X_in = Input(shape=(n_nodes, n_features))
    A_in = Input(shape=(n_nodes, n_nodes))

    # GCN Layer to capture structural information
    graph_conv = GCNConv(32, activation='relu')([X_in, A_in])
    graph_conv = GCNConv(16, activation='relu')([graph_conv, A_in])
    
    # Flatten and pass to Bi-LSTM for temporal patterns
    flattened = tf.keras.layers.Reshape((-1, 16))(graph_conv)
    bilstm = Bidirectional(LSTM(64, return_sequences=False))(flattened)
    
    dropout = Dropout(0.3)(bilstm)
    output = Dense(1, activation='linear')(dropout) # Predicting price/return

    model = tf.keras.Model(inputs=[X_in, A_in], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model