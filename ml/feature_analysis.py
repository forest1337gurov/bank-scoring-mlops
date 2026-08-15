import pandas as pd


class FeatureAnalyzer:

    @staticmethod
    def analyze(df: pd.DataFrame) -> pd.DataFrame:

        report = pd.DataFrame({
            "feature": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing": df.isna().sum().values,
            "missing_%": (df.isna().mean() * 100).round(2).values,
            "unique": df.nunique().values,
        })

        report["unique_%"] = (
            report["unique"] / len(df) * 100
        ).round(2)

        report["most_frequent"] = [
            df[col].mode().iloc[0] if not df[col].mode().empty else None
            for col in df.columns
        ]

        report["most_frequent_%"] = [
            round(df[col].value_counts(normalize=True).iloc[0] * 100, 2)
            for col in df.columns
        ]

        return report.sort_values("missing_%", ascending=False)