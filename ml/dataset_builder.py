from ml.extract import load_equifax_features
from ml.extract import load_client_features

from ml.preprocess import create_target
from ml.preprocess import preprocess_dataset


class DatasetBuilder:

    def build(self):

        equifax = load_equifax_features()

        clients = load_client_features()

        dataset = clients.merge(
            equifax,
            how="inner",
            left_on="ApplicationId",
            right_on="ec.ApplicationId"
        )

        dataset = create_target(dataset)

        dataset = preprocess_dataset(dataset)

        return dataset