import pytest
from safeds.data.tabular.containers import Table, TaggedTable


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            TaggedTable._from_table(
                Table(
                    {
                        "feature_a": [0, 1, 2],
                        "feature_b": [3, 4, 5],
                        "target": [6, 7, 8],
                    },
                ),
                "target",
            ),
            Table(
                {
                    "feature_a": [0, 1, 2],
                    "feature_b": [3, 4, 5],
                },
            ),
        ),
    ],
    ids=["normal"],
)
def test_should_remove_target_column(table: TaggedTable, expected: Table) -> None:
    new_table = table.remove_target_column()
    assert new_table.schema == expected.schema
    assert new_table == expected
