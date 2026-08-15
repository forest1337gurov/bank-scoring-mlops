from pathlib import Path

import pandas as pd

from config.clickhouse import get_client


SQL_DIR = Path(__file__).parent.parent / "sql"


def _read_sql(filename: str) -> str:
    with open(SQL_DIR / filename, "r", encoding="utf-8") as f:
        return f.read()


def load_equifax_features(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    client = get_client()
    query = _read_sql("feature_equifax.sql")
    if start_date and end_date:
        query = query.format(start_date=start_date, end_date=end_date)
    return client.query_df(query)


def load_client_features(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    client = get_client()
    query = _read_sql("feature_client.sql")
    if start_date and end_date:
        query = query.format(start_date=start_date, end_date=end_date)
    return client.query_df(query)