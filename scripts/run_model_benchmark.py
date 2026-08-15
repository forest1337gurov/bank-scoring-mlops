import pandas as pd
import mlflow

from ml.mlflow_setup import setup_mlflow
from ml.train_logistic import LogisticTrainer
from ml.train_catboost import CatBoostTrainer


def main():
    logistic_result = LogisticTrainer().train()
    catboost_result = CatBoostTrainer().train()

    results = pd.DataFrame([logistic_result, catboost_result])
    results = results[["model_name", "auc", "gini", "ks", "model_path"]]

    print("\nMODEL COMPARISON")
    print(results.to_string(index=False))

    results.to_csv("data/model_comparison.csv", index=False)
    print("\nSaved to data/model_comparison.csv")

    setup_mlflow()
    with mlflow.start_run(run_name="ModelBenchmark"):
        mlflow.log_artifact("data/model_comparison.csv")
        mlflow.log_metric("best_auc", float(results["auc"].max()))


if __name__ == "__main__":
    main()