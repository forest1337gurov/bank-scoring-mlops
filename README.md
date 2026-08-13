# Bank Scoring MLOps

Production-ready MLOps project for credit scoring.

## Project Goal

This project demonstrates the complete machine learning lifecycle for a bank credit scoring system:

- Data storage in PostgreSQL
- Model training pipeline
- MLflow experiment tracking
- Airflow orchestration
- FastAPI inference service
- Gradio demo interface
- Automatic model promotion
- Dynamic model reload
- Monitoring and data drift detection
- Docker deployment
- CI/CD with GitHub Actions

---

## Tech Stack

- Python 3.13
- FastAPI
- Gradio
- PostgreSQL
- SQLAlchemy
- LightGBM
- MLflow
- Apache Airflow
- Docker & Docker Compose
- GitHub Actions
- Pytest

---

## Project Structure

```text
app/            FastAPI application
ml/             Training pipeline
airflow/        DAGs
database/       SQL initialization
monitoring/     Drift and monitoring
tests/          Unit tests
docker/         Infrastructure
```

