from ml.dataset_builder import DatasetBuilder
from ml.feature_analysis import FeatureAnalyzer


def main():

    print("Building dataset...")

    dataset = DatasetBuilder().build()

    print("Dataset loaded.")

    report = FeatureAnalyzer.analyze(dataset)

    report.to_excel(
        "data/feature_report.xlsx",
        index=False,
    )

    print(report.head())

    print("\nFeature report saved:")
    print("data/feature_report.xlsx")


if __name__ == "__main__":
    main()