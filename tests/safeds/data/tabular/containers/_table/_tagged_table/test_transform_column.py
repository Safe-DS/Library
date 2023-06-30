import pytest
from safeds.data.tabular.containers import TaggedTable
from safeds.exceptions import UnknownColumnNameError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "column_name", "table_transformed"),
    [
        (
            TaggedTable({"feature_a": [1, 2, 3], "feature_b": [4, 5, 6], "target": [1, 2, 3]}, "target"),
            "feature_a",
            TaggedTable({"feature_a": [2, 4, 6], "feature_b": [4, 5, 6], "target": [1, 2, 3]}, "target"),
        ),
        (
            TaggedTable({"feature_a": [1, 2, 3], "feature_b": [4, 5, 6], "target": [1, 2, 3]}, "target"),
            "target",
            TaggedTable({"feature_a": [1, 2, 3], "feature_b": [4, 5, 6], "target": [2, 4, 6]}, "target"),
        ),
        (
            TaggedTable({"feature_a": [1, 2, 3], "b": [4, 5, 6], "target": [1, 2, 3]}, target_name="target", feature_names=["feature_a"]),
            "b",
            TaggedTable({"feature_a": [1, 2, 3], "b": [8, 10, 12], "target": [1, 2, 3]}, target_name="target", feature_names=["feature_a"]),
        ),
    ],
    ids=["transform_feature_column", "transform_target_column", "transform_column_that_is_neither"],
)
def test_should_transform_column(table: TaggedTable, column_name: str, table_transformed: TaggedTable) -> None:
    result = table.transform_column(column_name, lambda row: row.get_value(column_name) * 2)
    assert_that_tagged_tables_are_equal(result, table_transformed)


@pytest.mark.parametrize(
    ("table", "column_name"),
    [
        (
            TaggedTable(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["a", "b", "c"],
                },
                "C",
            ),
            "D",
        ),
        (
            TaggedTable(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["a", "b", "c"],
                },
                target_name="C",
                feature_names=["A"],
            ),
            "D"
        )
    ],
    ids=["has_only_features_and_target", "has_columns_that_are_neither"]
)
def test_should_raise_if_column_not_found(table: TaggedTable, column_name: str) -> None:
    with pytest.raises(UnknownColumnNameError, match=fr"Could not find column\(s\) '{column_name}'"):
        table.transform_column(column_name, lambda row: row.get_value("A") * 2)
