from fastapi import APIRouter

from app.services.scoring_service import scoring_service

router = APIRouter(prefix="/reload", tags=["Reload"])


@router.post("")
def reload_model():
    scoring_service.reload_model()
    return {"status": "ok", "message": "Model reloaded"}