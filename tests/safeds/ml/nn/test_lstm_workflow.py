from syrupy import SnapshotAssertion

from safeds.data.tabular.containers import TimeSeries, Table
from safeds.data.tabular.transformation import RangeScaler
from safeds.ml.nn import (
    ForwardLayer,
    InputConversionTimeSeries,
    NeuralNetworkRegressor,
    OutputConversionTimeSeries,
)

from tests.helpers import resolve_resource_path


def test_lstm_model(snapshot_png_image: SnapshotAssertion) -> None:
    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    time_series = TimeSeries.timeseries_from_csv_file(
        path=resolve_resource_path(_inflation_path),
        target_name="value",
        time_name="date",
    )
    test_values = Table.from_rows(time_series.to_rows()[-165:])._time_columns("value", "date")
    rs = RangeScaler()
    ss_2 = RangeScaler()
    ss_2 = ss_2.fit(time_series._as_table(), ["value"])
    time_series = rs.fit_and_transform(time_series._as_table(), ["value"])._time_columns(
        time_name=time_series.time.name,
        target_name=time_series.target.name,
        feature_names=time_series.features.column_names,
    )
    train_ts, test_ts = time_series.split_rows(0.8)

    model = NeuralNetworkRegressor(
        InputConversionTimeSeries(window_size=7, forecast_horizon=12),
        [ForwardLayer(input_size=7, output_size=256), ForwardLayer(input_size=256, output_size=1)],
        OutputConversionTimeSeries("predicted", window_size=7, forecast_horizon=12),
    )
    trained_model = model.fit(train_ts, epoch_size=25)

    pred_ts = trained_model.predict(test_ts)
    ss_2._column_names = ["predicted", "value"]
    ts = ((ss_2.inverse_transform(pred_ts._as_table().keep_only_columns(["predicted", "value"])).
          add_column(test_values.get_column("date"))).
          _time_columns("predicted", "date"))
    ts = ts.rename_column("value", "values")
    test_values = test_values.rename_column("value", "values")
    # suggest it ran through

    assert ts.plot_compare_time_series([test_values]) == snapshot_png_image
    assert ts.plot_lineplot() == snapshot_png_image
    assert test_values.plot_lineplot() == snapshot_png_image
    assert True
