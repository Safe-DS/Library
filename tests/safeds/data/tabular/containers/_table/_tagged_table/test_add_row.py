import pytest
from safeds.data.tabular.containers import Row, TaggedTable
from safeds.exceptions import UnknownColumnNameError

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
            TaggedTable({"feature": [], "target": []}, "target", ["feature"]),
            Row({"feat": None, "targ": None}),
            r"Could not find column\(s\) 'feature, target'",
        ),
    ],
    ids=["columns_missing"],
)
def test_should_raise_an_error_if_row_schema_invalid(
    tagged_table: TaggedTable,
    row: Row,
    error_msg: str,
) -> None:
    with pytest.raises(UnknownColumnNameError, match=error_msg):
        tagged_table.add_row(row)


@pytest.mark.parametrize(
    ("tagged_table", "row", "expected_table"),
    [
        (
            TaggedTable({"feature": [], "target": []}, "target"),
            Row({"feature": 2, "target": 5}),
            TaggedTable({"feature": [2], "target": [5]}, "target"),
        ),
    ],
    ids=["empty_feature_column"],
)
def test_should_add_row_to_empty_table(
    tagged_table: TaggedTable,
    row: Row,
    expected_table: TaggedTable,
) -> None:
    assert_that_tagged_tables_are_equal(tagged_table.add_row(row), expected_table)
