import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "result_train_table", "result_test_table"),
    [
        (Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         Table.from_dict({"col1": [1, 2], "col2": [1, 2]}),
         Table.from_dict({"col1": [1], "col2": [4]})),
    ],
    ids=["Table with three rows"]
)
def test_should_split_table(table, result_test_table, result_train_table) -> None:
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
    ids=["0.0%", "1.0%", "-1.0%", "2.0%"]
)
def test_should_raise_error(percentage_in_first: float) -> None:
    table = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})

    with pytest.raises(ValueError, match="the given percentage is not in range"):
        table.split(percentage_in_first)
