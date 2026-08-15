from ml.extract import load_client_features, load_equifax_features
from ml.inference import CatBoostPredictor
from ml.preprocess import preprocess_dataset


class ScoringService:
    def __init__(self) -> None:
        self.predictor = CatBoostPredictor()

    def score(self, application_id: int) -> dict:
        clients = load_client_features()
        equifax = load_equifax_features()

        dataset = clients.merge(
            equifax,
            how="inner",
            left_on="ApplicationId",
            right_on="ec.ApplicationId",
        )

        row = dataset[dataset["ApplicationId"] == application_id].copy()
        if row.empty:
            raise ValueError(f"ApplicationId {application_id} not found")

        row = preprocess_dataset(row)
        row = row.drop(columns=["Target"], errors="ignore")

        proba = self.predictor.predict_proba(row)

        if proba >= 0.5:
            risk = "High"
        elif proba >= 0.3:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "application_id": application_id,
            "default_probability": round(proba, 4),
            "risk_level": risk,
        }


scoring_service = ScoringService()