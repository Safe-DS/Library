import pytest

from safeds.data.tabular.containers import TaggedTable, Column, Table


@pytest.mark.parametrize(
    ("tagged_table", "column", "tagged_table_with_new_column"),
    [(
        Table({"f1": [1, 2], "target": [2, 3]}).tag_columns(target_name="target", feature_names=["f1"]),
        Column("f2", [4, 5]),
        Table({"f1": [1, 2], "target": [2, 3], "f2": [4, 5]}).tag_columns(target_name="target", feature_names=["f1", "f2"])
    ),(
        Table({"f1": [1, 2], "target": [2, 3], "other": [0, -1]}).tag_columns(target_name="target", feature_names=["f1"]),
        Column("f2", [4, 5]),
        Table({"f1": [1, 2], "target": [2, 3], "other": [0, -1], "f2": [4, 5]}).tag_columns(target_name="target", feature_names=["f1", "f2"])
    )], ids=["new column as feature", "table contains a non feature/target column"]
)
def test_add_column_as_feature(tagged_table: TaggedTable, column: Column, tagged_table_with_new_column) -> None:
    assert tagged_table.add_column_as_feature(column) == tagged_table_with_new_column
