import pandas as pd

from config.clickhouse import get_client
from ml.sql_loader import load_sql


def load_equifax_features() -> pd.DataFrame:
    """
    Загрузка агрегированных признаков Equifax.
    """

    client = get_client()
    query = load_sql("feature_equifax.sql")

    return client.query_df(query)


def load_client_features() -> pd.DataFrame:
    """
    Загрузка клиентских признаков.
    """

    client = get_client()
    query = load_sql("feature_client.sql")

    return client.query_df(query)