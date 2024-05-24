import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from safeds.ml.classical.classification import AdaBoostClassifier, BaselineClassifier
from safeds.ml.classical.regression import BaselineRegressor


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestBaselineRegressor:

    def test_workflow(self, training_set):
        input = Table.from_csv_file("D:\\Library_jetzt_aber_wirklich\\src\\safeds\\ml\\classical\\classification\\avocado.csv")
        table = input.remove_columns_except(["AveragePrice", "Total Volume", "4046", "4225", "4770", "Total Bags", "Small Bags", "Large Bags",
             "type"])
        [train, test] = table.split_rows(0.8)
        train = train.to_tabular_dataset(target_name="type")
        test = test.to_tabular_dataset(target_name="type")

        regressor = BaselineRegressor()
        fitted = regressor.fit(train)
        fitted.predict(test)
        assert fitted is not None

