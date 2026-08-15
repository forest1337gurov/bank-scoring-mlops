from ml.extract import load_equifax_features
from ml.extract import load_client_features

from ml.preprocess import create_target
from ml.preprocess import preprocess_dataset


class DatasetBuilder:

    def build(self):

        # Загружаем данные
        equifax = load_equifax_features()
        clients = load_client_features()

        # Объединяем
        dataset = clients.merge(
            equifax,
            how="inner",
            left_on="ApplicationId",
            right_on="ec.ApplicationId"
        )

        # Создаем таргет
        dataset = create_target(dataset)

        # Предобработка
        dataset = preprocess_dataset(dataset)

        return dataset