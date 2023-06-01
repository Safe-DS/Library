from safeds.data.tabular.containers import Row, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


def test_should_add_row() -> None:
    table = TaggedTable(
        {
            "feature": [0, 1],
            "target": [3, 4],
        },
        "target",
    )
    row = Row(
        {
            "feature": 2,
            "target": 5,
        },
    )
    new_table = table.add_row(row)
    expected = TaggedTable(
        {
            "feature": [0, 1, 2],
            "target": [3, 4, 5],
        },
        "target",
    )
    assert_that_tagged_tables_are_equal(new_table, expected)
