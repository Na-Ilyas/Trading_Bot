import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten

def create_cnn_lstm(input_shape):
    """
    Implementation of the CNN-LSTM architecture for price direction prediction.
    As per the paper, this model excels when paired with Boruta-selected features.
    """
    model = Sequential([
        # 1D Convolutional layer to capture local price/on-chain patterns
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM layer to model long-term temporal dependencies
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        
        # Output layer: 1 for 'Price Up', 0 for 'Price Down' (or 'Hold')
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model