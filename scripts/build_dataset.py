from pathlib import Path

import pandas as pd

from ml.dataset_builder import DatasetBuilder


def main():
    data_dir = Path("data")
    raw_client = pd.read_parquet(data_dir / "raw_client.parquet")
    raw_equifax = pd.read_parquet(data_dir / "raw_equifax.parquet")

    dataset = DatasetBuilder().build_from_dataframes(
        raw_client,
        raw_equifax,
        debug=True,
    )

    dataset_path = data_dir / "train_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)

    print("Saved:", dataset_path)
    print(dataset.shape)
    print(dataset["Target"].value_counts())


if __name__ == "__main__":
    main()