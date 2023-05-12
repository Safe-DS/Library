import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions._data import OutOfBoundsError


@pytest.mark.parametrize(
    ("table", "result_train_table", "result_test_table"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({"col1": [1, 2], "col2": [1, 2]}),
            Table({"col1": [1], "col2": [4]}),
        ),
    ],
    ids=["Table with three rows"],
)
def test_should_split_table(table: Table, result_test_table: Table, result_train_table: Table) -> None:
    train_table, test_table = table.split(2 / 3)
    assert result_test_table == test_table
    assert result_train_table == train_table


@pytest.mark.parametrize(
    "percentage_in_first",
    [
        0.0,
        1.0,
        -1.0,
        2.0,
    ],
    ids=["0.0%", "1.0%", "-1.0%", "2.0%"],
)
def test_should_raise_if_value_not_in_range(percentage_in_first: float) -> None:
    table = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})

    with pytest.raises(OutOfBoundsError, match=f"Value {percentage_in_first} is not in the range \\[0, 1\\]."):
        table.split(percentage_in_first)
