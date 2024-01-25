import pytest
from safeds.data.tabular.containers import Table
from torch.utils.data import DataLoader


@pytest.mark.parametrize(
    ("data", "target_name", "feature_names"),
    [
        (
            {
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 0],
            },
            "T",
            ["A", "B", "C"],
        ),
    ],
    ids=[
        "test",
    ],
)
def test_should_create_dataloader(
    data: dict[str, list[int]],
    target_name: str,
    feature_names: list[str] | None,
) -> None:
    tagged_table = Table.from_dict(data).tag_columns(target_name, feature_names)
    data_loader = tagged_table.into_dataloader(1)
    assert isinstance(data_loader, DataLoader)
