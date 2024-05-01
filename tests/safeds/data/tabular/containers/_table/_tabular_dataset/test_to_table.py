import pytest
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("tagged_table", "expected"),
    [
        (
            TabularDataset(
                {
                    "feature_1": [3, 9, 6],
                    "feature_2": [6, 12, 9],
                    "target": [1, 3, 2],
                },
                "target",
                ["feature_1", "feature_2"],
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
                ["feature_1", "feature_2"],
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
    ids=["normal", "table_with_column_as_non_feature"],
)
def test_should_return_table(tagged_table: TabularDataset, expected: Table) -> None:
    table = tagged_table.to_table()
    assert table.schema == expected.schema
    assert table == expected
