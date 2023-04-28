import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import DuplicateColumnNameError, UnknownColumnNameError


@pytest.mark.parametrize(
    ("name_from", "name_to", "column_one", "column_two"),
    [("A", "D", "D", "B"), ("A", "A", "A", "B")],
)
def test_rename_valid(name_from: str, name_to: str, column_one: str, column_two: str) -> None:
    table: Table = Table.from_dict({"A": [1], "B": [2]})
    renamed_table = table.rename_column(name_from, name_to)
    assert renamed_table.schema.has_column(column_one)
    assert renamed_table.schema.has_column(column_two)
    assert renamed_table.number_of_columns == 2


@pytest.mark.parametrize(
    ("name_from", "name_to", "error"),
    [
        ("C", "D", UnknownColumnNameError),
        ("A", "B", DuplicateColumnNameError),
        ("D", "D", UnknownColumnNameError),
    ],
)
def test_rename_invalid(name_from: str, name_to: str, error: Exception) -> None:
    table: Table = Table.from_dict({"A": [1], "B": [2]})
    with pytest.raises(error):
        table.rename_column(name_from, name_to)
