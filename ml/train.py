from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ml.dataset_builder import DatasetBuilder


class Trainer:

    def train(self):

        builder = DatasetBuilder()

        dataset = builder.build()

        X = dataset.drop(columns=["Target"])

        y = dataset["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model = LogisticRegression(
            max_iter=1000
        )

        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, proba)

        print(f"AUC = {auc:.4f}")

        return model