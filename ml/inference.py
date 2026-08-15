from pathlib import Path

import joblib
import pandas as pd


class CatBoostPredictor:
    def __init__(self, bundle_path: str | Path = "ml/artifacts/catboost_bundle.pkl") -> None:
        self.bundle_path = Path(bundle_path)
        self.load()

    def load(self) -> None:
        bundle = joblib.load(self.bundle_path)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]

    def reload(self) -> None:
        self.load()

    def predict_proba(self, df: pd.DataFrame) -> float:
        X = df.copy()
        X = X[self.feature_names]
        return float(self.model.predict_proba(X)[:, 1][0])