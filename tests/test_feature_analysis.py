from ml.dataset_builder import DatasetBuilder
from ml.feature_analysis import FeatureAnalyzer


def main():

    dataset = DatasetBuilder().build()

    report = FeatureAnalyzer.analyze(dataset)

    print(report)
    
    report.to_excel("data/feature_report.xlsx", index=False)

    print("Отчет сохранен в data/feature_report.xlsx")

if __name__ == "__main__":
    main()