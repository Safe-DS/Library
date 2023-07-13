import pytest
from safeds.data.tabular.containers import Row, Table, TaggedTable
from safeds.exceptions import SchemaMismatchError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "rows", "expected"),
    [
        (
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
            ),
        ),
    ],
    ids=["add_rows"],
)
def test_should_add_rows(table: TaggedTable, rows: list[Row], expected: TaggedTable) -> None:
    assert_that_tagged_tables_are_equal(table.add_rows(rows), expected)


@pytest.mark.parametrize(
    ("tagged_table", "rows", "error_msg"),
    [
        (
            TaggedTable({"feature": [1, 2, 3], "target": [4, 5, 6]}, "target", ["feature"]),
            [Row({"feature": 7, "target": 8}), Row({"feature": "a", "target": 9})],
            r"Failed because at least two schemas didn't match."
        ),
        (
            TaggedTable({"feature": [], "target": []}, "target", ["feature"]),
            [Row({"feat": None, "targ": None}), Row({"targ": None, "feat": None})],
            r"Failed because at least two schemas didn't match."
        ),
    ],
    ids=["invalid_schemas", "schemas_mismatch"],
)
def test_should_raise_an_error_if_rows_schemas_are_invalid(
    tagged_table: TaggedTable,
    rows: list[Row] | Table,
    error_msg: str,
) -> None:
    with pytest.raises(SchemaMismatchError, match=error_msg):
        tagged_table.add_rows(rows)
