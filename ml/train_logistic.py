import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset_builder import DatasetBuilder


class LogisticTrainer:

    @staticmethod
    def _calc_ks(y_true, proba) -> float:
        fpr, tpr, _ = roc_curve(y_true, proba)
        return float(max(tpr - fpr))

    def train(self):

        dataset = DatasetBuilder().build()

        X = dataset.drop(columns=["Target"])
        y = dataset["Target"]

        categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        numeric = X.select_dtypes(include=["number"]).columns.tolist()

        X[categorical] = X[categorical].astype(str)

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                ),
                (
                    "num",
                    StandardScaler(),
                    numeric,
                ),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, proba)
        gini = 2 * auc - 1
        ks = self._calc_ks(y_test, proba)

        print("=" * 50)
        print(f"AUC  = {auc:.4f}")
        print(f"Gini = {gini:.4f}")
        print(f"KS   = {ks:.4f}")

        joblib.dump(model, "ml/artifacts/logistic.pkl")
        print("Model saved.")

        return model