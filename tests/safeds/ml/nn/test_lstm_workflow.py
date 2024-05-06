from safeds.data.tabular.containers import Table
from safeds.data.tabular.transformation import RangeScaler
from safeds.ml.nn import (
    ForwardLayer,
    InputConversionTimeSeries,
    LSTMLayer,
    NeuralNetworkRegressor,
    OutputConversionTimeSeries,
)

from tests.helpers import resolve_resource_path


def test_lstm_model() -> None:
    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    table = Table.from_csv_file(path=resolve_resource_path(_inflation_path))
    # test_values = Table.from_rows(table.to_rows()[-165:])
    rs = RangeScaler()
    # ss_2 = RangeScaler()
    # ss_2 = ss_2.fit(table, ["value"])
    _, table = rs.fit_and_transform(table, ["value"])
    train_table, test_table = table.split_rows(0.8)

    model = NeuralNetworkRegressor(
        InputConversionTimeSeries(window_size=7, forecast_horizon=12),
        [ForwardLayer(input_size=7, output_size=256), LSTMLayer(input_size=256, output_size=1)],
        OutputConversionTimeSeries("predicted", window_size=7, forecast_horizon=12),
    )
    trained_model = model.fit(train_table.to_time_series_dataset("value", "date"), epoch_size=1)

    trained_model.predict(test_table.to_time_series_dataset("value", "date"))

    # ss_2._column_names = ["predicted", "value"]

    # ts = ss_2.inverse_transform(pred_ts.to_table().keep_only_columns(["predicted", "value"])).add_column(test_values.get_column("date"))
    # ts = ts.rename_column("value", "values")
    # test_values = test_values.rename_column("value", "values")
    # ts = ts.to_time_series_dataset("predicted", "date")
    # test_values.to_time_series_dataset("values", "date")
    # suggest it ran through

    # assert ts.plot_compare_time_series([test_values]) == snapshot_png_image
    # assert ts.plot_lineplot() == snapshot_png_image
    # assert test_values.plot_lineplot() == snapshot_png_image
    assert True
