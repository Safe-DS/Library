import pytest
from safeds.data.tabular.containers import Table
from torch.utils.data import DataLoader


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
def test_should_create_dataloader(
    data: dict[str, list[int]],
    target_name: str,
    extra_names: list[str] | None,
) -> None:
    tabular_dataset = Table.from_dict(data).to_tabular_dataset(target_name, extra_names)
    data_loader = tabular_dataset._into_dataloader_with_classes(1, 2)
    assert isinstance(data_loader, DataLoader)
