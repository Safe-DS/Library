from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import StandardScaler
from safeds.ml.nn import (
    ForwardLayer,
    InputConversionTable,
    NeuralNetworkRegressor,
    OutputConversionTable,
)

from tests.helpers import resolve_resource_path


def test_lstm_model() -> None:
    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    table_1 = Table.from_csv_file(
        path=resolve_resource_path(_inflation_path),
    )
    table_1 = table_1.remove_columns(["date"])
    table_2 = Table.from_rows(table_1.to_rows()[:-14])
    table_2 = table_2.add_columns([Table.from_rows(table_1.to_rows()[14:]).get_column("value").rename("target")])
    train_table, test_table = table_2.split_rows(0.8)

    ss = StandardScaler()
    _, train_table = ss.fit_and_transform(train_table, ["value"])
    _, test_table = ss.fit_and_transform(test_table, ["value"])
    model = NeuralNetworkRegressor(
        InputConversionTable(),
        [ForwardLayer(input_size=1, output_size=1)],
        OutputConversionTable("predicted"),
    )

    fitted_model = model.fit(train_table.to_tabular_dataset("target"), epoch_size=1, learning_rate=0.01)
    fitted_model.predict(test_table.keep_only_columns(["value"]))
    assert True