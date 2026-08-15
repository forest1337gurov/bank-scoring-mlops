from pathlib import Path

import joblib
import mlflow
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from ml.dataset_builder import DatasetBuilder
from ml.mlflow_setup import setup_mlflow


class CatBoostTrainer:
    def __init__(self) -> None:
        self.artifacts_dir = Path("ml/artifacts")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _calc_ks(y_true, proba) -> float:
        fpr, tpr, _ = roc_curve(y_true, proba)
        return float(max(tpr - fpr))

    def train(self, dataset=None):
        if dataset is None:
            dataset = DatasetBuilder().build()

        X = dataset.drop(columns=["Target"])
        y = dataset["Target"]

        categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        X[categorical] = X[categorical].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=100,
        )

        model.fit(
            X_train,
            y_train,
            cat_features=categorical,
            eval_set=(X_test, y_test),
            use_best_model=True,
        )

        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        gini = 2 * auc - 1
        ks = self._calc_ks(y_test, proba)

        print("=" * 50)
        print(f"AUC  = {auc:.4f}")
        print(f"Gini = {gini:.4f}")
        print(f"KS   = {ks:.4f}")

        model_path = self.artifacts_dir / "catboost.cbm"
        model.save_model(str(model_path))

        bundle_path = self.artifacts_dir / "catboost_bundle.pkl"
        joblib.dump(
            {
                "model": model,
                "feature_names": X.columns.tolist(),
                "categorical_features": categorical,
            },
            bundle_path,
        )

        setup_mlflow()
        with mlflow.start_run(run_name="CatBoost"):
            mlflow.log_params(
                {
                    "model": "catboost",
                    "iterations": 1000,
                    "learning_rate": 0.03,
                    "depth": 6,
                    "random_seed": 42,
                }
            )
            mlflow.log_metrics(
                {
                    "auc": auc,
                    "gini": gini,
                    "ks": ks,
                }
            )
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(bundle_path))

        print(f"Model saved: {model_path}")
        print(f"Bundle saved: {bundle_path}")

        return {
            "model_name": "CatBoost",
            "auc": auc,
            "gini": gini,
            "ks": ks,
            "model_path": str(model_path),
        }