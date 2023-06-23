import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import ColumnIsTargetError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "columns", "expected"),
    [
        (
            TaggedTable._from_table(Table({"feat1": [1, 2, 3], "feat2": [4, 5, 6], "target": [7, 8, 9]}), "target"),
            ["feat2"],
            TaggedTable._from_table(Table({"feat1": [1, 2, 3], "target": [7, 8, 9]}), "target"),
        ),
    ],
    ids=["only_features_remove_feature"],
)
def test_should_remove_columns(table: TaggedTable, columns: list[str], expected: TaggedTable) -> None:
    new_table = table.remove_columns(columns)
    assert_that_tagged_tables_are_equal(new_table, expected)


@pytest.mark.parametrize(
    ("table", "columns"),
    [(TaggedTable._from_table(Table({"feat": [1, 2, 3], "target": [4, 5, 6]}), "target"), ["target"])],
    ids=["only_features_and_target"],
)
def test_should_raise_column_is_target_error(table: TaggedTable, columns: list[str]) -> None:
    with pytest.raises(
        ColumnIsTargetError,
        match=r'Illegal schema modification: Column "target" is the target column and cannot be removed.',
    ):
        table.remove_columns(columns)
