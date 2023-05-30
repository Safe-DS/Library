import pytest
from _pytest.python_api import raises
from safeds.data.tabular.containers import TaggedTable
from safeds.exceptions import IndexOutOfBoundsError


@pytest.mark.parametrize(
    ("table", "test_table", "second_test_table"),
    [
        (
            TaggedTable({"feature": [1, 2, 1], "target": [1, 2, 4]}, "target", None),
            TaggedTable({"feature": [1, 2], "target": [1, 2]}, "target", None),
            TaggedTable({"feature": [1, 1], "target": [1, 4]}, "target", None),
        ),
    ],
    ids=["Table with three rows"],
)
def test_should_slice_rows(table: TaggedTable, test_table: TaggedTable, second_test_table: TaggedTable) -> None:
    new_table = table.slice_rows(0, 2, 1)
    second_new_table = table.slice_rows(0, 3, 2)
    third_new_table = table.slice_rows()
    assert new_table.schema == test_table.schema
    assert new_table.features == test_table.features
    assert new_table.target == test_table.target
    assert new_table == test_table
    assert second_new_table.schema == second_test_table.schema
    assert second_new_table.features == second_test_table.features
    assert second_new_table.target == second_test_table.target
    assert second_new_table == second_test_table
    assert third_new_table.schema == table.schema
    assert third_new_table.features == table.features
    assert third_new_table.target == table.target
    assert third_new_table == table


@pytest.mark.parametrize(
    ("start", "end", "step", "error_message"),
    [
        (3, 2, 1, r"There is no element in the range \[3, 2\]"),
        (4, 0, 1, r"There is no element in the range \[4, 0\]"),
        (0, 4, 1, r"There is no element at index '4'"),
        (-4, 0, 1, r"There is no element at index '-4'"),
        (0, -4, 1, r"There is no element in the range \[0, -4\]"),
    ],
)
def test_should_raise_if_index_out_of_bounds(start: int, end: int, step: int, error_message: str) -> None:
    table = TaggedTable({"feature": [1, 2, 1], "target": [1, 2, 4]}, "target", None)

    with raises(IndexOutOfBoundsError, match=error_message):
        table.slice_rows(start, end, step)
