import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.classification import AdaBoostClassifier, BaselineClassifier


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestBaselineClassifier:

    def test_workflow(self, training_set):
        classifier = BaselineClassifier()
        fitted = classifier.fit(training_set)
        fitted.predict(training_set)
        assert fitted is not None

