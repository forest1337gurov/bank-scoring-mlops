from pathlib import Path
import pandas as pd
from ml.train_catboost import CatBoostTrainer

def main():
    dataset = pd.read_parquet(Path("data") / "train_dataset.parquet")
    result = CatBoostTrainer().train(dataset=dataset)
    print(result)

if __name__ == "__main__":
    main()