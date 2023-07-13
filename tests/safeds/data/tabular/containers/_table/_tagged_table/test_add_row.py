import pytest
from safeds.data.tabular.containers import Row, TaggedTable
from safeds.exceptions import SchemaMismatchError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "row", "expected"),
    [
        (
            TaggedTable(
                {
                    "feature": [0, 1],
                    "target": [3, 4],
                },
                "target",
            ),
            Row(
                {
                    "feature": 2,
                    "target": 5,
                },
            ),
            TaggedTable(
                {
                    "feature": [0, 1, 2],
                    "target": [3, 4, 5],
                },
                "target",
            ),
        ),
    ],
    ids=["add_row"],
)
def test_should_add_row(table: TaggedTable, row: Row, expected: TaggedTable) -> None:
    assert_that_tagged_tables_are_equal(table.add_row(row), expected)


@pytest.mark.parametrize(
    ("tagged_table", "row", "error_msg"),
    [
        (
            TaggedTable({"feature": [1, 2, 3], "target": [4, 5, 6]}, "target", ["feature"]),
            Row({"feature": "a", "target": 8}),
            r"Failed because at least two schemas didn't match.",
        ),
        (
            TaggedTable({"feature": [], "target": []}, "target", ["feature"]),
            Row({"feat": None, "targ": None}),
            r"Failed because at least two schemas didn't match.",
        ),
    ],
    ids=["invalid_schemas", "schemas_mismatch"],
)
def test_should_raise_an_error_if_row_schema_invalid(
    tagged_table: TaggedTable,
    row: Row,
    error_msg: str,
) -> None:
    with pytest.raises(SchemaMismatchError, match=error_msg):
        tagged_table.add_row(row)
