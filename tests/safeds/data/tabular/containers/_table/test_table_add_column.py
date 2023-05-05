import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.exceptions import ColumnSizeError, DuplicateColumnNameError


@pytest.mark.parametrize(
    ("input_table", "expected", "column"),
    [
        (Table.from_dict({"A": [1, 3, 5], "B": [2, 4, 6]}),
         Table.from_dict({"A": [1, 3, 5], "B": [2, 4, 6], "C": ["a", "b", "c"]}),
         Column("C", ["a", "b", "c"]),
         )
    ],
    ids=["Column with characters"],
)
def test_should_add_column(input_table: Table, expected: Table, column: Column) -> None:
    result = input_table.add_column(column)
    assert expected == result


@pytest.mark.parametrize(
    ("column_values", "column_name", "error"),
    [
        (["a", "b", "c"], "B", DuplicateColumnNameError),
        (["a", "b"], "C", ColumnSizeError),
    ],
    ids=["Duplicate Column Name Error", "Column Size Error"]
)
def test_should_raise_error(column_values: list[str], column_name: str, error: type[Exception]) -> None:
    input_table = Table.from_dict(
        {
            "A": [1, 3, 5],
            "B": [2, 4, 6],
        },
    )
    column = Column(column_name, column_values)

    with pytest.raises(error):
        input_table.add_column(column)
