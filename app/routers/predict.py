from fastapi import APIRouter, HTTPException

from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.scoring_service import scoring_service

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        return scoring_service.score(payload.application_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))