from pathlib import Path
import pandas as pd
from ml.train_logistic import LogisticTrainer

def main():
    dataset = pd.read_parquet(Path("data") / "train_dataset.parquet")
    result = LogisticTrainer().train(dataset=dataset)
    print(result)

if __name__ == "__main__":
    main()