import pytest
from safeds._config import _get_device
from safeds.data.tabular.containers import Table
from torch.types import Device
from torch.utils.data import DataLoader

from tests.helpers import configure_test_with_device, get_devices, get_devices_ids


@pytest.mark.parametrize(
    ("data", "target_name", "extra_names"),
    [
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "T",
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
    extra_names: list[str] | None,
    device: Device,
) -> None:
    configure_test_with_device(device)
    tabular_dataset = Table.from_dict(data).to_tabular_dataset(target_name, extra_names=extra_names)
    data_loader = tabular_dataset._into_dataloader_with_classes(1, 2)
    batch = next(iter(data_loader))
    assert batch[0].device == _get_device()
    assert batch[1].device == _get_device()
    assert isinstance(data_loader, DataLoader)
