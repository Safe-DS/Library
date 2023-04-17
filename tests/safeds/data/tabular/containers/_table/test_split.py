import pytest

from safeds.data.tabular.containers import Table


def test_split_valid() -> None:
    table = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    result_train_table = Table.from_dict({"col1": [1, 2], "col2": [1, 2]})
    result_test_table = Table.from_dict({"col1": [1], "col2": [4]})
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
)
def test_split_invalid(percentage_in_first: float) -> None:
    table = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})

    with pytest.raises(ValueError, match="the given percentage is not in range"):
        table.split(percentage_in_first)
