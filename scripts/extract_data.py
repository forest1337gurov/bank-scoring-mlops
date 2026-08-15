from datetime import date
from pathlib import Path
import os

from ml.extract import load_client_features, load_equifax_features


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Чтобы DAG работал стабильно, задаём окно явно.
    # Потом это можно вынести в .env.
    start_date = os.getenv("TRAIN_START_DATE", "2025-06-01")
    end_date = os.getenv("TRAIN_END_DATE", "2025-08-01")

    clients = load_client_features(start_date=start_date, end_date=end_date)
    equifax = load_equifax_features(start_date=start_date, end_date=end_date)

    clients.to_parquet(data_dir / "raw_client.parquet", index=False)
    equifax.to_parquet(data_dir / "raw_equifax.parquet", index=False)

    print("Saved:")
    print(data_dir / "raw_client.parquet")
    print(data_dir / "raw_equifax.parquet")


if __name__ == "__main__":
    main()