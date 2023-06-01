import pytest
from _pytest.python_api import raises
from safeds.data.tabular.containers import TaggedTable
from safeds.exceptions import IndexOutOfBoundsError

from tests.helpers import assert_that_tagged_tables_are_equal


@pytest.mark.parametrize(
    ("table", "test_table", "second_test_table"),
    [
        (
            TaggedTable({"feature": [1, 2, 1], "target": [1, 2, 4]}, "target"),
            TaggedTable({"feature": [1, 2], "target": [1, 2]}, "target"),
            TaggedTable({"feature": [1, 1], "target": [1, 4]}, "target"),
        ),
    ],
    ids=["Table with three rows"],
)
def test_should_slice_rows(table: TaggedTable, test_table: TaggedTable, second_test_table: TaggedTable) -> None:
    new_table = table.slice_rows(0, 2, 1)
    second_new_table = table.slice_rows(0, 3, 2)
    third_new_table = table.slice_rows()
    assert_that_tagged_tables_are_equal(new_table, test_table)
    assert_that_tagged_tables_are_equal(second_new_table, second_test_table)
    assert_that_tagged_tables_are_equal(third_new_table, table)


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
    table = TaggedTable({"feature": [1, 2, 1], "target": [1, 2, 4]}, "target")

    with raises(IndexOutOfBoundsError, match=error_message):
        table.slice_rows(start, end, step)
