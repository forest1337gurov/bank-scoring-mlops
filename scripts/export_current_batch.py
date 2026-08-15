from datetime import date, timedelta
from pathlib import Path

from ml.dataset_builder import DatasetBuilder
from ml.extract import load_client_features, load_equifax_features


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Свежий batch: последние 7 дней
    end_date = date.today()
    start_date = end_date - timedelta(days=7)

    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    clients = load_client_features(start_date=start_date_str, end_date=end_date_str)
    equifax = load_equifax_features(start_date=start_date_str, end_date=end_date_str)

    dataset = DatasetBuilder().build_features_only_from_dataframes(
        clients,
        equifax,
        debug=True,
    )

    current_batch_path = data_dir / "current_batch.parquet"
    dataset.to_parquet(current_batch_path, index=False)

    print(f"Saved to {current_batch_path}")
    print(dataset.shape)


if __name__ == "__main__":
    main()