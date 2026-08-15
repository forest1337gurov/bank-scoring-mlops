from ml.dataset_builder import DatasetBuilder


def main():

    builder = DatasetBuilder()

    dataset = builder.build()

    print("=" * 60)
    print("Размер датасета:")
    print(dataset.shape)

    print("=" * 60)
    print(dataset.head())

    print("=" * 60)
    print(dataset["Target"].value_counts())


if __name__ == "__main__":
    main()