import pytest

from safeds.data.tabular.containers import Row, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "rows", "expected"),
    [(
        TaggedTable(
            {
                "feature": [0, 1],
                "target": [4, 5],
            },
            "target",
        ),
        [
            Row(
                {
                    "feature": 2,
                    "target": 6,
                },
            ),
            Row({"feature": 3, "target": 7}),
        ],
        TaggedTable(
            {
                "feature": [0, 1, 2, 3],
                "target": [4, 5, 6, 7],
            },
            "target",
        )
    )],
    ids=["add_rows"]
)
def test_should_add_rows(table: TaggedTable, rows: list[Row], expected: TaggedTable) -> None:
    assert_that_tagged_tables_are_equal(table.add_rows(rows), expected)
