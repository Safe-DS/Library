from safeds.data.tabular.containers import TaggedTable, Table
from safeds.ml.regression import Regressor


class DummyRegressor(Regressor):
    """
    Dummy regressor to test metrics.

    Metrics methods expect a `TaggedTable` as input with two columns:

    - `predicted`: The predicted targets.
    - `expected`: The correct targets.

    `target_name` must be set to `"expected"`.
    """

    def fit(self, training_set: TaggedTable) -> None:
        pass

    def predict(self, dataset: Table) -> TaggedTable:
        # Needed until https://github.com/Safe-DS/Stdlib/issues/75 is fixed
        predicted = dataset.get_column("predicted")
        feature = predicted.rename("feature")
        dataset = Table.from_columns([feature, predicted])

        return TaggedTable(dataset, target_name="predicted")
