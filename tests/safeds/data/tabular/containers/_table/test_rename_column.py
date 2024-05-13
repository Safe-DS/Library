import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError, DuplicateColumnError


@pytest.mark.parametrize(
    ("name_from", "name_to", "column_one", "column_two"),
    [("A", "D", "D", "B"), ("A", "A", "A", "B")],
    ids=["column renamed", "column not renamed"],
)
def test_should_rename_column(name_from: str, name_to: str, column_one: str, column_two: str) -> None:
    table: Table = Table({"A": [1], "B": [2]})
    renamed_table = table.rename_column(name_from, name_to)
    assert renamed_table.schema.column_names == [column_one, column_two]
    assert renamed_table.column_names == [column_one, column_two]


@pytest.mark.parametrize("table", [Table({"A": [1], "B": [2]}), Table()], ids=["normal", "empty"])
def test_should_raise_if_old_column_does_not_exist(table: Table) -> None:
    with pytest.raises(ColumnNotFoundError):
        table.rename_column("C", "D")


def test_should_raise_if_new_column_exists_already() -> None:
    table: Table = Table({"A": [1], "B": [2]})
    with pytest.raises(DuplicateColumnError):
        table.rename_column("A", "B")
