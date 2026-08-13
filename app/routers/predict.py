from fastapi import APIRouter

from app.schemas.credit_application import CreditApplication
from app.schemas.prediction import PredictionResponse
from app.services.prediction_service import prediction_service

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.post(
    "",
    response_model=PredictionResponse,
)
def predict(application: CreditApplication):
    return prediction_service.predict(application)