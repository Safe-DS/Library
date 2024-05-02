import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("tabular_dataset", "expected"),
    [
        (
            TabularDataset(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
                "target",
            ),
            Table(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
            ),
        ),
        (
            TabularDataset(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "other": [3, 9, 12],
                    "target": [1, 3, 2],
                },
                "target",
                ["other"],
            ),
            Table(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "other": [3, 9, 12],
                    "target": [1, 3, 2],
                },
            ),
        ),
    ],
    ids=["normal", "table_with_extra_column"],
)
def test_should_return_table(tabular_dataset: TabularDataset, expected: Table) -> None:
    table = tabular_dataset.to_table()
    assert table.schema == expected.schema
    assert table == expected
