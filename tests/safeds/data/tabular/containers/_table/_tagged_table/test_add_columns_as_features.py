import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("tagged_table", "columns", "tagged_table_with_new_columns"),
    [
        (
            Table({"f1": [1, 2], "target": [2, 3]}).tag_columns(target_name="target", feature_names=["f1"]),
            [Column("f2", [4, 5]), Column("f3", [6, 7])],
            Table({"f1": [1, 2], "target": [2, 3], "f2": [4, 5], "f3": [6, 7]}).tag_columns(
                target_name="target",
                feature_names=["f1", "f2", "f3"],
            ),
        ),
        (
            Table({"f1": [1, 2], "target": [2, 3]}).tag_columns(target_name="target", feature_names=["f1"]),
            Table.from_columns([Column("f2", [4, 5]), Column("f3", [6, 7])]),
            Table({"f1": [1, 2], "target": [2, 3], "f2": [4, 5], "f3": [6, 7]}).tag_columns(
                target_name="target",
                feature_names=["f1", "f2", "f3"],
            ),
        ),
        (
            Table({"f1": [1, 2], "target": [2, 3], "other": [0, -1]}).tag_columns(
                target_name="target",
                feature_names=["f1"],
            ),
            Table.from_columns([Column("f2", [4, 5]), Column("f3", [6, 7])]),
            Table({"f1": [1, 2], "target": [2, 3], "other": [0, -1], "f2": [4, 5], "f3": [6, 7]}).tag_columns(
                target_name="target",
                feature_names=["f1", "f2", "f3"],
            ),
        ),
    ],
    ids=["new columns as feature", "table added as features", "table contains a non feature/target column"],
)
def test_add_columns_as_features(
    tagged_table: TaggedTable,
    columns: list[Column] | Table,
    tagged_table_with_new_columns: TaggedTable,
) -> None:
    assert_that_tagged_tables_are_equal(tagged_table.add_columns_as_features(columns), tagged_table_with_new_columns)


@pytest.mark.parametrize(
    ("tagged_table", "columns", "error_msg"),
    [
        (
            TaggedTable({"A": [1, 2, 3], "B": [4, 5, 6]}, target_name="B", feature_names=["A"]),
            [Column("A", [7, 8, 9]), Column("D", [10, 11, 12])],
            r"Column 'A' already exists.",
        ),
    ],
    ids=["column_already_exist"],
)
def test_add_columns_raise_duplicate_column_name_if_column_already_exist(
    tagged_table: TaggedTable,
    columns: list[Column] | Table,
    error_msg: str,
) -> None:
    with pytest.raises(DuplicateColumnNameError, match=error_msg):
        tagged_table.add_columns_as_features(columns)


@pytest.mark.parametrize(
    ("tagged_table", "columns", "error_msg"),
    [
        (
            TaggedTable({"A": [1, 2, 3], "B": [4, 5, 6]}, target_name="B", feature_names=["A"]),
            [Column("C", [5, 7, 8, 9]), Column("D", [4, 10, 11, 12])],
            r"Expected a column of size 3 but got column of size 4.",
        ),
    ],
    ids=["columns_are_oversize"],
)
def test_should_raise_column_size_error_if_columns_are_oversize(
    tagged_table: TaggedTable,
    columns: list[Column] | Table,
    error_msg: str,
) -> None:
    with pytest.raises(ColumnSizeError, match=error_msg):
        tagged_table.add_columns_as_features(columns)
