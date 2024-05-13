import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table1", "expected"),
    [
        (Table({"col1": ["col1_1"], "col2": ["col2_1"]}), Column("col1", ["col1_1"])),
    ],
    ids=["First column"],
)
def test_should_get_column(table1: Table, expected: Column) -> None:
    assert table1.get_column("col1") == expected


@pytest.mark.parametrize(
    "table",
    [
        (Table({"col1": ["col1_1"], "col2": ["col2_1"]})),
        (Table()),
    ],
    ids=["no col3", "empty"],
)
def test_should_raise_error_if_column_name_unknown(table: Table) -> None:
    with pytest.raises(ColumnNotFoundError):
        table.get_column("col3")
