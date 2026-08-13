from app.schemas.credit_application import CreditApplication
from app.schemas.prediction import PredictionResponse


class PredictionService:
    """
    Сервис для получения скорингового предсказания.
    Пока используется простая демонстрационная логика.
    Позже сюда будет подключена ML-модель.
    """

    def predict(self, application: CreditApplication) -> PredictionResponse:
        score = application.credit_score
        dti = application.debt_to_income

        probability = (
            (850 - score) / 550 * 0.7 +
            dti * 0.3
        )

        probability = max(0.0, min(probability, 1.0))

        if probability < 0.30:
            risk = "Low"

        elif probability < 0.60:
            risk = "Medium"

        else:
            risk = "High"

        return PredictionResponse(
            default_probability=round(probability, 3),
            risk_level=risk
        )


prediction_service = PredictionService()