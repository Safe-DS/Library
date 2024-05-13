import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "result_train_table", "result_test_table", "percentage_in_first"),
    [
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({"col1": [1, 2], "col2": [4, 2]}),
            Table({"col1": [1], "col2": [1]}),
            2 / 3,
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({"col1": [], "col2": []}),
            Table({"col1": [1, 2, 1], "col2": [4, 2, 1]}),
            0,
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            Table({"col1": [1, 2, 1], "col2": [4, 2, 1]}),
            Table({"col1": [], "col2": []}),
            1,
        ),
    ],
    ids=["2/3%", "0%", "100%"],
)
def test_should_split_table(
    table: Table,
    result_test_table: Table,
    result_train_table: Table,
    percentage_in_first: int,
) -> None:
    #test if schema stayed the same
    schema = table.schema
    train_table, test_table = table.split_rows(percentage_in_first)
    assert result_test_table == test_table
    assert schema == train_table.schema
    assert result_train_table == train_table


@pytest.mark.parametrize(
    "percentage_in_first",
    [
        -1.0,
        2.0,
    ],
    ids=["-100%", "200%"],
)
def test_should_raise_if_value_not_in_range(percentage_in_first: float) -> None:
    table = Table({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    with pytest.raises(ValueError, match=r"is not inside \[0, 1\]"):
        table.split_rows(percentage_in_first)


def test_should_split_empty_table() -> None:
    t1, t2 = Table().split_rows(0.4)
    assert t1.number_of_rows == 0
    assert t2.number_of_rows == 0
