import pytest
from safeds._config import _get_device
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError
from torch.types import Device
from torch.utils.data import DataLoader
import torch

from tests.helpers import configure_test_with_device, get_devices, get_devices_ids


@pytest.mark.parametrize(
    ("data", "target_name", "time_name", "extra_names"),
    [
        (
            {
                "A": [1, 4, 3],
                "B": [2, 5, 4],
                "C": [3, 6, 5],
                "T": [0, 1, 6],
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
@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_should_create_dataloader(
    data: dict[str, list[int]],
    target_name: str,
    time_name: str,
    extra_names: list[str] | None,
    device: Device,
) -> None:
    configure_test_with_device(device)
    tabular_dataset = Table.from_dict(data).to_time_series_dataset(target_name, time_name, extra_names)
    data_loader = tabular_dataset._into_dataloader_with_window(1, 1, 1)
    batch = next(iter(data_loader))
    assert batch[0].device == _get_device()
    assert batch[1].device == _get_device()
    assert isinstance(data_loader, DataLoader)


@pytest.mark.parametrize(
    ("data", "target_name", "time_name", "extra_names"),
    [
        (
            {
                "A": [1, 4, 3],
                "B": [2, 5, 4],
                "C": [3, 6, 5],
                "T": [0, 1, 6],
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
@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_should_create_dataloader_predict(
    data: dict[str, list[int]],
    target_name: str,
    time_name: str,
    extra_names: list[str] | None,
    device: Device,
) -> None:
    configure_test_with_device(device)
    tabular_dataset = Table.from_dict(data).to_time_series_dataset(target_name, time_name, extra_names)
    data_loader = tabular_dataset._into_dataloader_with_window_predict(1, 1, 1)
    batch = next(iter(data_loader))
    assert batch[0].device == _get_device()
    assert isinstance(data_loader, DataLoader)


@pytest.mark.parametrize(
    ("data", "window_size", "forecast_horizon", "error_type", "error_msg"),
    [
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ).to_time_series_dataset("T", "B"),
            1,
            2,
            ValueError,
            r"Can not create windows with window size less then forecast horizon \+ window_size",
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ).to_time_series_dataset("T", "B"),
            1,
            0,
            OutOfBoundsError,
            None,
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ).to_time_series_dataset("T", "B"),
            0,
            1,
            OutOfBoundsError,
            None,
        ),
    ],
    ids=[
        "forecast_and_window",
        "forecast",
        "window_size",
    ],
)
@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_should_create_dataloader_invalid(
    data: TimeSeriesDataset,
    window_size: int,
    forecast_horizon: int,
    error_type: type[ValueError],
    error_msg: str | None,
    device: Device,
) -> None:
    configure_test_with_device(device)
    with pytest.raises(error_type, match=error_msg):
        data._into_dataloader_with_window(window_size=window_size, forecast_horizon=forecast_horizon, batch_size=1)


@pytest.mark.parametrize(
    ("data", "window_size", "forecast_horizon", "error_type", "error_msg"),
    [
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ).to_time_series_dataset("T", "B"),
            1,
            2,
            ValueError,
            r"Can not create windows with window size less then forecast horizon \+ window_size",
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ).to_time_series_dataset("T", "B"),
            1,
            0,
            OutOfBoundsError,
            None,
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ).to_time_series_dataset("T", "B"),
            0,
            1,
            OutOfBoundsError,
            None,
        ),
    ],
    ids=[
        "forecast_and_window",
        "forecast",
        "window_size",
    ],
)
@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
def test_should_create_dataloader_predict_invalid(
    data: TimeSeriesDataset,
    window_size: int,
    forecast_horizon: int,
    error_type: type[ValueError],
    error_msg: str | None,
    device: Device,
) -> None:
    configure_test_with_device(device)
    with pytest.raises(error_type, match=error_msg):
        data._into_dataloader_with_window_predict(
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            batch_size=1,
        )

def test_continues_dataloader() -> None:
    ts = Table(
        {"a": [1, 2, 3, 4, 5, 6, 7], "b": [1, 2, 3, 4, 5, 6, 7], "c": [1, 2, 3, 4, 5, 6, 7]}
    ).to_time_series_dataset("a", "b")
    dl = ts._into_dataloader_with_window(1, 2, 1, True)
    dl_2 = ts._into_dataloader_with_window(1, 2, 1, False)
    assert len(dl_2.dataset.Y) == len(dl.dataset.Y)
    # 4mal 2er Arrays mit 1er Einträgen
    assert dl.dataset.Y.shape == torch.Size([4, 2, 1])
