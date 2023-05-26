import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import DuplicateColumnNameError, UnknownColumnNameError


@pytest.mark.parametrize(
    ("name_from", "name_to", "column_one", "column_two"),
    [("A", "D", "D", "B"), ("A", "A", "A", "B")],
    ids=["column renamed", "column not renamed"],
)
def test_should_rename_column(name_from: str, name_to: str, column_one: str, column_two: str) -> None:
    table: Table = Table({"A": [1], "B": [2]})
    renamed_table = table.rename_column(name_from, name_to)
    assert renamed_table.schema.has_column(column_one)
    assert renamed_table.schema.has_column(column_two)
    assert renamed_table.number_of_columns == 2


@pytest.mark.parametrize(
    "table",
    [
        Table({"A": [1], "B": [2]}),
        Table()
    ], ids=["normal", "empty"])
def test_should_raise_if_old_column_does_not_exist(table: Table) -> None:
    with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'C'"):
        table.rename_column("C", "D")


def test_should_raise_if_new_column_exists_already() -> None:
    table: Table = Table({"A": [1], "B": [2]})
    with pytest.raises(DuplicateColumnNameError, match=r"Column 'B' already exists."):
        table.rename_column("A", "B")
