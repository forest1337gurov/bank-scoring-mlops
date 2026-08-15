from datetime import datetime

from airflow.decorators import dag
from airflow.operators.bash import BashOperator


@dag(
    schedule="0 6 * * 1",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["bank-scoring", "retrain", "mlflow"],
)
def weekly_retrain():
    extract_data = BashOperator(
        task_id="extract_data",
        bash_command="cd /opt/airflow/project && python -m scripts.extract_data",
    )

    build_dataset = BashOperator(
        task_id="build_dataset",
        bash_command="cd /opt/airflow/project && python -m scripts.build_dataset",
    )

    train_logreg = BashOperator(
        task_id="train_logreg",
        bash_command="cd /opt/airflow/project && python -m scripts.run_train_parquet",
    )

    train_catboost = BashOperator(
        task_id="train_catboost",
        bash_command="cd /opt/airflow/project && python -m scripts.run_train_catboost_parquet",
    )

    benchmark = BashOperator(
        task_id="benchmark_models",
        bash_command="cd /opt/airflow/project && python -m scripts.run_model_benchmark_parquet",
    )

    decide_promotion = BashOperator(
        task_id="decide_promotion",
        bash_command="cd /opt/airflow/project && python -m scripts.decide_promotion",
    )

    promote_model = BashOperator(
        task_id="promote_model",
        bash_command="cd /opt/airflow/project && python -m scripts.promote_model",
    )

    skip_promotion = BashOperator(
        task_id="skip_promotion",
        bash_command='echo "Candidate is not better than production. Skip promotion."',
    )

    reload_api = BashOperator(
        task_id="reload_api",
        bash_command="cd /opt/airflow/project && python -m scripts.reload_api",
    )

    export_current_batch = BashOperator(
        task_id="export_current_batch",
        bash_command="cd /opt/airflow/project && python -m scripts.export_current_batch",
    )

    monitoring = BashOperator(
        task_id="monitoring",
        bash_command="cd /opt/airflow/project && python -m scripts.run_monitoring",
    )

    extract_data >> build_dataset >> [train_logreg, train_catboost] >> benchmark >> decide_promotion
    decide_promotion >> promote_model >> reload_api >> export_current_batch >> monitoring
    decide_promotion >> skip_promotion >> export_current_batch


weekly_retrain()