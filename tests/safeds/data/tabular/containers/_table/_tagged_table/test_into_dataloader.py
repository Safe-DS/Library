import pytest
from torch.utils.data import DataLoader

from safeds.data.tabular.containers import Table


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
    # Todo Fix that TaggedTable doesnt have a split_rows Method

    tagged_table = Table.from_dict(data).tag_columns(target_name, feature_names)
    data_loader = tagged_table.into_dataloader(1)
    assert isinstance(data_loader, DataLoader)
