import pandas as pd


class EDA:

    @staticmethod
    def info(df: pd.DataFrame):


        print("Размер датасета")
        print(df.shape)

        print("Типы данных")
        print(df.dtypes)

        print("Пропуски")
        print(df.isna().sum().sort_values(ascending=False))

        print("Target")
        print(df["Target"].value_counts(normalize=True))