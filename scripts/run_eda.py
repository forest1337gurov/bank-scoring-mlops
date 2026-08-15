from ml.dataset_builder import DatasetBuilder
from ml.eda import EDA


def main():

    dataset = DatasetBuilder().build()

    EDA.info(dataset)


if __name__ == "__main__":
    main()