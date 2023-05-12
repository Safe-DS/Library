import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions._data import OutOfBoundsError


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
    new_table = table.slice_rows(0, 2, 1)
    second_new_table = table.slice_rows(0, 3, 2)
    third_new_table = table.slice_rows()
    assert new_table == test_table
    assert second_new_table == second_test_table
    assert third_new_table == table


@pytest.mark.parametrize(
    ("start", "end", "step"),
    [(3, 2, 1)],
)
def test_should_raise_value_error_if_start_larger_than_end(start: int, end: int, step: int) -> None:
    table = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})

    with pytest.raises(ValueError, match="The given end index is smaller than the given start index"):
        table.slice_rows(start, end, step)


@pytest.mark.parametrize(
    ("start", "end", "step"),
    [(-5, 2, 1)],
)
def test_should_raise_out_of_bounds_error_if_start_smaller_than_zero(start: int, end: int, step: int) -> None:
    table = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})

    with pytest.raises(OutOfBoundsError, match=f"Value {start} is not in the range \\[0, 3\\]."):
        table.slice_rows(start, end, step)


@pytest.mark.parametrize(
    ("start", "end", "step"),
    [(2, 4, 1)],
)
def test_should_raise_out_of_bounds_error_if_end_is_larger_than_number_of_rows(start: int, end: int, step: int) -> None:
    table = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})

    with pytest.raises(OutOfBoundsError, match=f"Value {end} is not in the range \\[0, 3\\]."):
        table.slice_rows(start, end, step)
