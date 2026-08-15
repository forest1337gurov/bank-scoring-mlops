from datetime import datetime
from pathlib import Path

from airflow.decorators import dag
from airflow.operators.bash import BashOperator


PROJECT_ROOT = Path("/opt/airflow/project")


@dag(
    schedule="0 6 * * 1",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["bank-scoring", "retrain"],
)
def weekly_retrain():
    run_benchmark = BashOperator(
        task_id="run_model_benchmark",
        bash_command=f"cd {PROJECT_ROOT} && python -m scripts.run_model_benchmark",
    )

    run_benchmark


weekly_retrain()