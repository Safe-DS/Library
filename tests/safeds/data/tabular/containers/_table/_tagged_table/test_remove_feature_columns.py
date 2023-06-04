import pytest
from safeds.data.tabular.containers import TaggedTable
from safeds.exceptions import UnknownColumnNameError, ColumnIsTaggedError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "column_names", "expected"),
    [
        (
            TaggedTable(
                {
                    "feature_1": [0, 1],
                    "feature_2": [2, 3],
                    "target": [4, 5],
                },
                "target",
            ),
            ["feature_1"],
            TaggedTable(
                {
                    "feature_2": [2, 3],
                    "target": [4, 5],
                },
                "target",
            ),
        ),
        (
            TaggedTable(
                {
                    "feature_1": [0, 1],
                    "feature_2": [2, 3],
                    "target": [4, 5],
                },
                "target",
            ),
            [],
            TaggedTable(
                {
                    "feature_1": [0, 1],
                    "feature_2": [2, 3],
                    "target": [4, 5],
                },
                "target",
            ),
        ),
    ],
    ids=["remove_some", "remove_none"]
)
def test_should_remove_columns(table: TaggedTable, column_names: list[str], expected: TaggedTable) -> None:
    new_table = table.remove_feature_columns(column_names)
    assert_that_tagged_tables_are_equal(new_table, expected)

@pytest.mark.parametrize(
    ("table", "column_names"),
    [
        (
            TaggedTable(
                {
                    "feature_1": [0, 1],
                    "feature_2": [2, 3],
                    "target": [4, 5],
                },
                "target",
            ),
            ["feature_3"],
        ),
    ],
    ids=["remove_unknown"]
)
def test_should_throw_unknown_columns(table: TaggedTable, column_names: list[str]) -> None:
    with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'feature_3'"):
        table.remove_feature_columns(column_names)

@pytest.mark.parametrize(
    ("table", "column_names"),
    [
        (
            TaggedTable(
                {
                    "feature_1": [0, 1],
                    "feature_2": [2, 3],
                    "target": [4, 5],
                },
                "target",
            ),
            ["feature_1", "feature_2"],
        ),
    ],
    ids=["remove_all_features"]
)
def test_should_throw_no_features_left(table: TaggedTable, column_names: list[str]) -> None:
    with pytest.raises(ValueError, match="At least one feature column must be specified."):
        table.remove_feature_columns(column_names)

@pytest.mark.parametrize(
    ("table", "column_names",),
    [
        (
            TaggedTable(
                {
                    "feature_1": [0, 1],
                    "feature_2": [2, 3],
                    "target": [4, 5],
                },
                "target",
            ),
            ["target"],
        ),
    ],
    ids=["remove_target"]
)
def test_should_throw_column_is_tagged(table: TaggedTable, column_names: list[str]) -> None:
    with pytest.raises(ColumnIsTaggedError,
                       match='Illegal schema modification: Column "target" is tagged and cannot be removed.'):
        table.remove_feature_columns(column_names)
