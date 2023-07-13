import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("tagged_table", "column", "tagged_table_with_new_column"),
    [
        (
            Table({"f1": [1, 2], "target": [2, 3]}).tag_columns(target_name="target", feature_names=["f1"]),
            Column("f2", [4, 5]),
            Table({"f1": [1, 2], "target": [2, 3], "f2": [4, 5]}).tag_columns(
                target_name="target",
                feature_names=["f1", "f2"],
            ),
        ),
        (
            Table({"f1": [1, 2], "target": [2, 3], "other": [0, -1]}).tag_columns(
                target_name="target",
                feature_names=["f1"],
            ),
            Column("f2", [4, 5]),
            Table({"f1": [1, 2], "target": [2, 3], "other": [0, -1], "f2": [4, 5]}).tag_columns(
                target_name="target",
                feature_names=["f1", "f2"],
            ),
        ),
    ],
    ids=["new column as feature", "table contains a non feature/target column"],
)
def test_should_add_column_as_feature(
    tagged_table: TaggedTable,
    column: Column,
    tagged_table_with_new_column: TaggedTable,
) -> None:
    assert_that_tagged_tables_are_equal(tagged_table.add_column_as_feature(column), tagged_table_with_new_column)


@pytest.mark.parametrize(
    ("tagged_table", "column", "error_msg"),
    [
        (
            TaggedTable({"A": [1, 2, 3], "B": [4, 5, 6]}, target_name="B", feature_names=["A"]),
            Column("A", [7, 8, 9]),
            r"Column 'A' already exists.",
        ),
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
            TaggedTable({"A": [1, 2, 3], "B": [4, 5, 6]}, target_name="B", feature_names=["A"]),
            Column("C", [5, 7, 8, 9]),
            r"Expected a column of size 3 but got column of size 4.",
        ),
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
