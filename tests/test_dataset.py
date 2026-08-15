from ml.dataset_builder import DatasetBuilder


def main():
    builder = DatasetBuilder()

    dataset = builder.build()

    print("Размер датасета:")
    print(dataset.shape)

    print("Первые 5 строк:")
    print(dataset.head())

    print("Target:")
    print(dataset["Target"].value_counts())


if __name__ == "__main__":
    main()