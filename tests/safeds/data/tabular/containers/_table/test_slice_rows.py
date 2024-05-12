import pytest
from _pytest.python_api import raises
from safeds.data.tabular.containers import Table
from safeds.exceptions import IndexOutOfBoundsError


@pytest.mark.parametrize(
    ("table", "test_table", "second_test_table"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({"col1": [1, 2], "col2": [1, 2]}),
            Table({"col1": [1, 1], "col2": [1, 4]}),
        ),
    ],
    ids=["Table with three rows"],
)
def test_should_slice_rows(table: Table, test_table: Table, second_test_table: Table) -> None:
    new_table = table.slice_rows(0, 2)
    second_new_table = table.slice_rows(0, 3)
    third_new_table = table.slice_rows()
    assert new_table.schema == test_table.schema
    assert new_table == test_table
    assert second_new_table.schema == second_test_table.schema
    assert second_new_table == second_test_table
    assert third_new_table.schema == table.schema
    assert third_new_table == table  # TODO: parameterize this test

# TODO: there's now another interface
# @pytest.mark.parametrize(
#     ("start", "end", "step", "error_message"),
#     [
#         (3, 2, 1, r"There is no element in the range \[3, 2\]"),
#         (4, 0, 1, r"There is no element in the range \[4, 0\]"),
#         (0, 4, 1, r"There is no element at index '4'"),
#         (-4, 0, 1, r"There is no element at index '-4'"),
#         (0, -4, 1, r"There is no element in the range \[0, -4\]"),
#     ],
#     ids=["Start > End", "Start > Length", "End > Length", "Start < 0", "End < 0"],
# )
# def test_should_raise_if_index_out_of_bounds(start: int, end: int, step: int, error_message: str) -> None:
#     table = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
#
#     with raises(IndexOutOfBoundsError, match=error_message):
#         table.slice_rows(start, end)


def test_should_raise_if_index_out_of_bounds_on_empty() -> None:
    table = Table()

    with pytest.raises(IndexOutOfBoundsError, match="There is no element at index '2'"):
        table.slice_rows(2, 5)
