import mlflow

def setup_mlflow() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bank-scoring")