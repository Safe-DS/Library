import pytest
from safeds.data.tabular.containers import TaggedTable
from safeds.exceptions import ColumnIsTaggedError


def test_should_remove_column() -> None:
    table = TaggedTable(
        {
            "feature_numerical": [0, 1, 2],
            "feature_non_numerical": ["a", "b", "c"],
            "target": [3, 4, 5],
        },
        "target",
    )
    new_table = table.remove_columns_with_non_numerical_values()
    expected = TaggedTable(
        {
            "feature_numerical": [0, 1, 2],
            "target": [3, 4, 5],
        },
        "target",
    )
    assert new_table.schema == expected.schema
    assert new_table.features == expected.features
    assert new_table.target == expected.target
    assert new_table == expected


def test_should_throw_column_is_tagged() -> None:
    table = TaggedTable(
        {
            "feature": [0, 1, 2],
            "target": ["a", "b", "c"],
        },
        "target",
    )
    with pytest.raises(
        ColumnIsTaggedError,
        match='Illegal schema modification: Column "target" is tagged and cannot be removed.',
    ):
        table.remove_columns_with_non_numerical_values()
