from safeds.data.tabular.containers import Column, TaggedTable

from tests.helpers import assert_that_tagged_tables_are_equal


def test_should_add_columns() -> None:
    table = TaggedTable(
        {
            "feature_1": [0, 1, 2],
            "target": [3, 4, 5],
        },
        "target",
    )
    cols = [
        Column("other", [6, 7, 8]),
        Column("other2", [9, 6, 3]),
    ]
    new_table = table.add_columns(cols)
    expected = TaggedTable(
        {
            "feature_1": [0, 1, 2],
            "target": [3, 4, 5],
            "other": [6, 7, 8],
            "other2": [9, 6, 3],
        },
        "target",
        ["feature_1"],
    )
    assert_that_tagged_tables_are_equal(new_table, expected)
