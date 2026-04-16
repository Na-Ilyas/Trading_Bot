"""
Feature Selection Engine
- Boruta (wrapper) and LightGBM (embedded) feature importance
- Applied STRICTLY on training data to prevent leakage
- Returns the intersection or union of selected features
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import config as C

warnings.filterwarnings("ignore")


def boruta_select(X_train: pd.DataFrame, y_train: np.ndarray) -> list:
    """Boruta feature selection on training data only."""
    from boruta import BorutaPy

    # Validate input
    if X_train.isnull().any().any():
        print("  [Boruta] WARNING: NaN values detected, filling with 0")
        X_clean = X_train.fillna(0).values
    else:
        X_clean = X_train.values

    rf = RandomForestClassifier(
        n_estimators=100, n_jobs=-1,
        random_state=C.SYNTHETIC_SEED,
        max_depth=C.BORUTA_RF_DEPTH,
    )
    selector = BorutaPy(
        rf, n_estimators="auto",
        max_iter=C.BORUTA_MAX_ITER,
        random_state=C.SYNTHETIC_SEED,
        verbose=2
    )
    selector.fit(X_clean, y_train)

    selected = X_train.columns[selector.support_].tolist()
    # Also include "tentative" features (borderline important)
    tentative = X_train.columns[selector.support_weak_].tolist()

    print(f"  [Boruta] Confirmed: {len(selected)}, Tentative: {len(tentative)}")

    # Print top-10 feature importances for debugging
    importances = pd.Series(selector.estimator_.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    print("  [Boruta] Top-10 feature importances:")
    for feat, imp in importances.head(10).items():
        print(f"    {feat}: {imp:.4f}")

    return selected + tentative


def lightgbm_select(X_train: pd.DataFrame, y_train: np.ndarray) -> list:
    """LightGBM feature importance → top-K features."""
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, num_leaves=31,
        random_state=C.SYNTHETIC_SEED, verbose=-1
    )
    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    top_k = min(C.LGBM_TOP_K, len(importances))
    selected = importances.head(top_k).index.tolist()
    print(f"  [LightGBM] Top-{top_k} features selected")
    return selected


def run_feature_selection(X_train: pd.DataFrame, y_train: np.ndarray) -> list:
    """
    Run both methods and return UNION of selected features.
    This ensures we don't miss anything either method finds important.
    """
    all_selected = set()

    if C.USE_BORUTA:
        try:
            boruta_feats = boruta_select(X_train, y_train)
            all_selected.update(boruta_feats)
        except Exception as e:
            print(f"  [Boruta] Failed: {e}, falling back to LightGBM only")

    if C.USE_LIGHTGBM:
        lgbm_feats = lightgbm_select(X_train, y_train)
        all_selected.update(lgbm_feats)

    selected = sorted(all_selected)

    if len(selected) == 0:
        print("  [WARNING] No features selected, using all features")
        selected = X_train.columns.tolist()

    print(f"  [FINAL] {len(selected)} features selected (union)")
    return selected
