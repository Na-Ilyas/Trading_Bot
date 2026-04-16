"""
Baseline Models for Comparative Analysis
═════════════════════════════════════════
StandaloneLSTM, SimpleRNN, XGBoost, ARIMA, BuyAndHold
All share a common interface: fit(X_train, y_train, X_val, y_val) and predict(X_test).
"""

import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import config as C


class BaselineModel:
    """Common interface for all baseline models."""
    name = "BaselineModel"

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        """Train the model. Returns a history dict with at least 'train_time'."""
        raise NotImplementedError

    def predict(self, X_test) -> np.ndarray:
        """Return probabilities (N,) in [0, 1]."""
        raise NotImplementedError


class StandaloneLSTM(BaselineModel):
    """2-layer LSTM with BatchNorm, no GCN branch."""
    name = "LSTM"

    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        tf.random.set_seed(C.SYNTHETIC_SEED)
        t0 = time.time()
        units = C.LSTM_STANDALONE_UNITS
        inp = Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = LSTM(units[0], return_sequences=True)(inp)
        x = BatchNormalization()(x)
        x = LSTM(units[1])(x)
        x = BatchNormalization()(x)
        x = Dropout(C.DROPOUT_RATE)(x)
        x = Dense(32, activation="relu")(x)
        out = Dense(1, activation="sigmoid")(x)

        self.model = Model(inp, out)
        self.model.compile(optimizer=Adam(learning_rate=C.LEARNING_RATE),
                           loss="binary_crossentropy", metrics=["accuracy"])

        val_data = (X_val, y_val) if X_val is not None else None
        history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE,
            callbacks=[EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PAT,
                                     restore_best_weights=True, verbose=0)],
            verbose=0
        )
        return {"train_time": time.time() - t0, "history": history.history}

    def predict(self, X_test) -> np.ndarray:
        return self.model.predict(X_test, verbose=0).flatten()


class SimpleRNNModel(BaselineModel):
    """2-layer SimpleRNN baseline."""
    name = "RNN"

    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        tf.random.set_seed(C.SYNTHETIC_SEED)
        t0 = time.time()
        units = C.RNN_UNITS
        inp = Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = SimpleRNN(units[0], return_sequences=True)(inp)
        x = SimpleRNN(units[1])(x)
        x = Dropout(C.DROPOUT_RATE)(x)
        x = Dense(32, activation="relu")(x)
        out = Dense(1, activation="sigmoid")(x)

        self.model = Model(inp, out)
        self.model.compile(optimizer=Adam(learning_rate=C.LEARNING_RATE),
                           loss="binary_crossentropy", metrics=["accuracy"])

        val_data = (X_val, y_val) if X_val is not None else None
        history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE,
            callbacks=[EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PAT,
                                     restore_best_weights=True, verbose=0)],
            verbose=0
        )
        return {"train_time": time.time() - t0, "history": history.history}

    def predict(self, X_test) -> np.ndarray:
        return self.model.predict(X_test, verbose=0).flatten()


class XGBoostModel(BaselineModel):
    """XGBoost classifier on flattened windows."""
    name = "XGBoost"

    def __init__(self):
        self.model = None

    def _flatten(self, X):
        """Flatten 3D (samples, timesteps, features) to 2D."""
        return X.reshape(len(X), -1)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        from xgboost import XGBClassifier

        t0 = time.time()
        self.model = XGBClassifier(
            n_estimators=C.XGB_N_ESTIMATORS,
            max_depth=C.XGB_MAX_DEPTH,
            learning_rate=0.05,
            random_state=C.SYNTHETIC_SEED,
            eval_metric="logloss",
            verbosity=0,
        )

        X_flat = self._flatten(X_train)
        eval_set = [(self._flatten(X_val), y_val)] if X_val is not None else None
        self.model.fit(X_flat, y_train, eval_set=eval_set, verbose=False)
        return {"train_time": time.time() - t0}

    def predict(self, X_test) -> np.ndarray:
        return self.model.predict_proba(self._flatten(X_test))[:, 1]


class ARIMAModel(BaselineModel):
    """ARIMA baseline on close prices. Predicts direction as 0 or 1."""
    name = "ARIMA"

    def __init__(self):
        self.train_prices = None
        self.order = C.ARIMA_ORDER

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        t0 = time.time()
        # X_train is 3D windowed; extract close prices from last timestep
        # We store raw close prices passed separately
        self.train_prices = X_train
        return {"train_time": time.time() - t0}

    def predict_from_prices(self, train_close: np.ndarray, n_predict: int) -> np.ndarray:
        """ARIMA prediction using raw close prices."""
        from statsmodels.tsa.arima.model import ARIMA

        t0 = time.time()
        probs = np.full(n_predict, 0.5)

        # Use rolling ARIMA: fit on history, predict 1-step ahead
        history = list(train_close[-500:])  # use last 500 for speed
        for i in range(n_predict):
            try:
                model = ARIMA(history, order=self.order)
                fit = model.fit()
                forecast = fit.forecast(steps=1)[0]
                current = history[-1]
                probs[i] = 1.0 if forecast > current else 0.0
            except Exception:
                probs[i] = 0.5
            # Append actual (we don't have it here, so append forecast as proxy)
            history.append(history[-1])  # keep history length stable

        self._train_time = time.time() - t0
        return probs

    def predict(self, X_test) -> np.ndarray:
        # Fallback: return 0.5 if called without prices
        return np.full(len(X_test), 0.5)


class BuyAndHoldModel(BaselineModel):
    """Always predicts UP (prob=1.0). Pure long strategy baseline."""
    name = "BuyAndHold"

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        return {"train_time": 0.0}

    def predict(self, X_test) -> np.ndarray:
        return np.ones(len(X_test))


def get_all_baselines() -> list:
    """Returns list of all baseline model instances."""
    return [
        StandaloneLSTM(),
        SimpleRNNModel(),
        XGBoostModel(),
        ARIMAModel(),
        BuyAndHoldModel(),
    ]
