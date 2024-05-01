import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("tagged_table", "target_column"),
    [
        (
            TabularDataset(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
                target_name="T",
            ),
            Column("T", [0, 1]),
        ),
    ],
    ids=["target"],
)
def test_should_return_target(tagged_table: TabularDataset, target_column: Column) -> None:
    assert tagged_table.target == target_column
