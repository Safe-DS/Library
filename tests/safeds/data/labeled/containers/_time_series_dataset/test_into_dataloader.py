import pytest
from safeds.data.tabular.containers import Table
from safeds.data.labeled.containers import TimeSeriesDataset
from torch.utils.data import DataLoader


@pytest.mark.parametrize(
    ("data", "target_name", "time_name", "extra_names"),
    [
        (
            {
                "A": [1, 4,3],
                "B": [2, 5,4],
                "C": [3, 6,5],
                "T": [0, 1,6],
            },
            "T",
            "B",
            [],
        ),
    ],
    ids=[
        "test",
    ],
)
def test_should_create_dataloader(
    data: dict[str, list[int]],
    target_name: int,
    time_name: int,
    extra_names: list[str] | None,
) -> None:
    tabular_dataset = Table.from_dict(data).to_time_series_dataset(target_name, time_name, extra_names)
    data_loader = tabular_dataset._into_dataloader_with_window(1, 1, 1)
    assert isinstance(data_loader, DataLoader)

@pytest.mark.parametrize(
    ("data", "window_size", "forecast_horizon", "error_type", "error_msg"),
    [
        (
            Table({
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            }).to_time_series_dataset("T", "B"),
            1,
            2,
            ValueError,
            r'Can not create windows with window size less then forecast horizon \+ window_size',
        ),
        (
            Table({
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            }).to_time_series_dataset("T", "B"),
            1,
            0,
            ValueError,
            r"forecast_horizon must be greater than or equal to 1",
        ),
        (
            Table({
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            }).to_time_series_dataset("T", "B"),
            0,
            1,
            ValueError,
            r"window_size must be greater than or equal to 1",
        ),
    ],
    ids=[
        "forecast_and_window",
        "forecast",
        "window_size",
    ],
)
def test_should_create_dataloader_invalid(
    data: TimeSeriesDataset,
    window_size: str,
    forecast_horizon: str,
    error_type: ValueError,
    error_msg: str,
) -> None:
    with pytest.raises(error_type, match = error_msg):
        data._into_dataloader_with_window(window_size=window_size, forecast_horizon=forecast_horizon, batch_size=1)
