import pytest
from safeds.data.labeled.containers import TaggedTable
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table", "target_name", "feature_names", "error", "error_msg"),
    [
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            ["A", "B", "C", "D", "E"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D, E'",
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "D",
            ["A", "B", "C"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D'",
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "A",
            ["A", "B", "C"],
            ValueError,
            r"Column 'A' cannot be both feature and target.",
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "A",
            [],
            ValueError,
            r"At least one feature column must be specified.",
        ),
        (
            Table(
                {
                    "A": [1, 4],
                },
            ),
            "A",
            None,
            ValueError,
            r"At least one feature column must be specified.",
        ),
    ],
    ids=[
        "feature_does_not_exist",
        "target_does_not_exist",
        "target_and_feature_overlap",
        "features_are_empty-explicitly",
        "features_are_empty_implicitly",
    ],
)
def test_should_raise_error(
    table: Table,
    target_name: str,
    feature_names: list[str] | None,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error, match=error_msg):
        TaggedTable._from_table(table, target_name=target_name, feature_names=feature_names)


@pytest.mark.parametrize(
    ("table", "target_name", "feature_names"),
    [
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            ["A", "B", "C"],
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            ["A", "C"],
        ),
        (
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            None,
        ),
    ],
    ids=["create_tagged_table", "tagged_table_not_all_columns_are_features", "tagged_table_with_feature_names_as_None"],
)
def test_should_create_a_tagged_table(table: Table, target_name: str, feature_names: list[str] | None) -> None:
    tagged_table = TaggedTable._from_table(table, target_name=target_name, feature_names=feature_names)
    feature_names = feature_names if feature_names is not None else table.remove_columns([target_name]).column_names
    assert isinstance(tagged_table, TaggedTable)
    assert tagged_table._features.column_names == feature_names
    assert tagged_table._target.name == target_name
    assert tagged_table._features == table.keep_only_columns(feature_names)
    assert tagged_table._target == table.get_column(target_name)
