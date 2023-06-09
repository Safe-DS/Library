import pytest
from safeds.data.tabular.containers import TaggedTable
from safeds.exceptions import ColumnIsTargetError

from tests.helpers import assert_that_tagged_tables_are_equal


def test_should_remove_column() -> None:
    table = TaggedTable(
        {
            "feature_complete": [0, 1, 2],
            "feature_incomplete": [3, None, 5],
            "target": [6, 7, 8],
        },
        "target",
    )
    new_table = table.remove_columns_with_missing_values()
    expected = TaggedTable(
        {
            "feature_complete": [0, 1, 2],
            "target": [6, 7, 8],
        },
        "target",
    )
    assert_that_tagged_tables_are_equal(new_table, expected)


def test_should_throw_column_is_target() -> None:
    table = TaggedTable(
        {
            "feature": [0, 1, 2],
            "target": [3, None, 5],
        },
        "target",
    )
    with pytest.raises(
        ColumnIsTargetError,
        match='Illegal schema modification: Column "target" is the target column and cannot be removed.',
    ):
        table.remove_columns_with_missing_values()
