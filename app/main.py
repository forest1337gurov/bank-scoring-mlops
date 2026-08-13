from fastapi import FastAPI

from app.core.config import settings
from app.routers.health import router as health_router
from app.routers.predict import router as predict_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
)

app.include_router(health_router)
app.include_router(predict_router)


@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}"
    }