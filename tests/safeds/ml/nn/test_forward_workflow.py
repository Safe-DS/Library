from typing import Any

import pytest
from safeds.data.tabular.containers import Table, TaggedTable

from safeds.ml.nn import (
    ForwardLayer,
    InputConversionTable,
    NeuralNetworkRegressor,
    OutputConversionTable,
    LSTMLayer,
)

from tests.helpers import resolve_resource_path


def test_lstm_model() -> None:
    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    table_1 = Table.from_csv_file(
        path=resolve_resource_path(_inflation_path),
    )
    table_2 = Table.from_rows(table_1.to_rows()[:-14])
    table_2 = table_2.add_column(Table.from_rows(table_1.to_rows()[14:]).get_column("value").rename("target"))
    train_table, test_table = table_2.split_rows(0.8)
    model = NeuralNetworkRegressor(
        InputConversionTable(["value"], "target"),
        [ForwardLayer(input_size=1, output_size=256), ForwardLayer(input_size=256, output_size=1)],
        OutputConversionTable("predicted"),
    )
    fitted_model = model.fit(train_table.tag_columns("target", ["value"]), epoch_size=25)
    predictions = fitted_model.predict(test_table.keep_only_columns(["value"]))
    print(predictions)
    assert False
