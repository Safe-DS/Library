import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable
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
