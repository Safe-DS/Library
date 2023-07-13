import pytest
from safeds.data.tabular.containers import Column, TaggedTable
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("tagged_table", "column", "expected_tagged_table"),
    [
        (
            TaggedTable(
                {
                    "feature_1": [0, 1, 2],
                    "target": [3, 4, 5],
                },
                "target",
                None,
            ),
            Column("other", [6, 7, 8]),
            TaggedTable(
                {
                    "feature_1": [0, 1, 2],
                    "target": [3, 4, 5],
                    "other": [6, 7, 8],
                },
                "target",
                ["feature_1"],
            ),
        ),
    ],
    ids=["add_column_as_non_feature"],
)
def test_should_add_column(tagged_table: TaggedTable, column: Column, expected_tagged_table: TaggedTable) -> None:
    assert_that_tagged_tables_are_equal(tagged_table.add_column(column), expected_tagged_table)


@pytest.mark.parametrize(
    ("tagged_table", "column", "error_msg"),
    [
        (
            TaggedTable({"A": ["a", "b", "c"], "B": ["d", "e", "f"]}, target_name="B", feature_names=["A"]),
            Column("B", ["g", "h", "i"]),
            r"Column 'B' already exists."
        )
    ],
    ids=["column_already_exists"],
)
def test_should_raise_duplicate_column_name_if_column_already_exists(
    tagged_table: TaggedTable,
    column: Column,
    error_msg: str,
) -> None:
    with pytest.raises(DuplicateColumnNameError, match=error_msg):
        tagged_table.add_column_as_feature(column)


@pytest.mark.parametrize(
    ("tagged_table", "column", "error_msg"),
    [
        (
            TaggedTable({"A": ["a", "b", "c"], "B": ["d", "e", "f"]}, target_name="B", feature_names=["A"]),
            Column("C", ["g", "h", "i", "j"]),
            r"Expected a column of size 3 but got column of size 4."
        )
    ],
    ids=["column_is_oversize"],
)
def test_should_raise_column_size_error_if_column_is_oversize(
    tagged_table: TaggedTable,
    column: Column,
    error_msg: str,
) -> None:
    with pytest.raises(ColumnSizeError, match=error_msg):
        tagged_table.add_column_as_feature(column)
