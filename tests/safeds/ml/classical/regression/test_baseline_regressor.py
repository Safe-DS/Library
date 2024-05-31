import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import StandardScaler
from safeds.ml.classical.regression import BaselineRegressor


@pytest.fixture()
def training_set() -> TabularDataset:
    table = Table({"col1": [1, 2, 3, 4], "col2": [1, 2, 3, 4]})
    return table.to_tabular_dataset(target_name="col1")


class TestBaselineRegressor:

    def test_workflow(self, training_set):
        import time
        input = Table.from_csv_file("D:\\Library_jetzt_aber_wirklich\\src\\safeds\\ml\\classical\\regression\\houses.csv")
        table = input.remove_columns(["id", "lat", "long", "zipcode", "condition", "grade", "date"])
        #TODO Not scaling the data makes the Regressor take 10 Minutes instead of 20 Seconds
        target = table.get_column("price")
        ss = StandardScaler(column_names=table.column_names.remove("price"))
        [_, scaled_features] = ss.fit_and_transform(table.remove_columns(["price"]))
        table = scaled_features.add_columns([target])

        [train, test] = table.split_rows(0.8)
        train = train.to_tabular_dataset(target_name="price")
        test = test.to_tabular_dataset(target_name="price")

        start_time = time.time()
        regressor = BaselineRegressor(include_slower_models=False)
        fitted = regressor.fit(train)
        results = fitted.predict(test)
        end_time = time.time()

        print(f"Time needed: {end_time-start_time}")
        assert fitted is not None

