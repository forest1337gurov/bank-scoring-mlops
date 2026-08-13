from pydantic import BaseModel


class PredictionResponse(BaseModel):
    default_probability: float
    risk_level: str