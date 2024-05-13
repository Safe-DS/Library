import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table", "column_names", "expected"),
    [
        (
            Table({"A": [1], "B": [2]}),
            [],
            Table(),
        ),
        (
            Table({"A": [1], "B": [2]}),
            ["A"],
            Table({"A": [1]}),
        ),
        (
            Table({"A": [1], "B": [2]}),
            ["B"],
            Table({"B": [2]}),
        ),
        (
            Table({"A": [1], "B": [2]}),
            ["A", "B"],
            Table({"A": [1], "B": [2]}),
        ),
        # Related to https://github.com/Safe-DS/Library/issues/115
        (
            Table({"A": [1], "B": [2], "C": [3]}),
            ["C", "A"],
            Table({"C": [3], "A": [1]}),
        ),
        (
            Table(),
            [],
            Table(),
        ),
    ],
    ids=["No Column Name", "First Column", "Second Column", "All columns", "Last and first columns", "empty"],
)
def test_should_remove_all_except_listed_columns(table: Table, column_names: list[str], expected: Table) -> None:
    transformed_table = table.remove_columns_except(column_names)
    assert transformed_table.schema == expected.schema
    assert transformed_table == expected
    if len(column_names) == 0:
        assert expected.number_of_rows == 0


@pytest.mark.parametrize("table", [Table({"A": [1], "B": [2]}), Table()], ids=["table", "empty"])
def test_should_raise_error_if_column_name_unknown(table: Table) -> None:
    with pytest.raises(ColumnNotFoundError, match=r"Could not find column\(s\) 'C'"):
        table.remove_columns_except(["C"])
