import numpy as np
import pandas as pd


def create_target(df: pd.DataFrame) -> pd.DataFrame:

    dataset = df.copy()

    dataset["Target"] = np.where(
        dataset["MaxOverdueDays90"] >= 25,
        1,
        0,
    )

    dataset = dataset.drop(
        columns=["MaxOverdueDays90"]
    )

    return dataset


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:

    dataset = df.copy()

    dataset = dataset.drop(
        columns=[
            "ApplicationId",
            "ec.ApplicationId"
        ],
        errors="ignore"
    )

    dataset = dataset.fillna(0)

    return dataset