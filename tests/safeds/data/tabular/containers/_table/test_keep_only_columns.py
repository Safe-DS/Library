import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table", "column_names", "expected"),
    [
        (
            Table({"A": [1], "B": [2]}),
            [],
            Table({}),
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
        # Related to https://github.com/Safe-DS/Stdlib/issues/115
        (
            Table({"A": [1], "B": [2], "C": [3]}),
            ["C", "A"],
            Table({"C": [3], "A": [1]}),
        ),
    ],
    ids=["No Column Name", "First Column", "Second Column", "All columns", "Last and first columns"],
)
def test_should_keep_only_listed_columns(table: Table, column_names: list[str], expected: Table) -> None:
    transformed_table = table.keep_only_columns(column_names)
    assert transformed_table == expected


def test_should_raise_error_if_column_name_unknown() -> None:
    table = Table({"A": [1], "B": [2]})
    with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'C'"):
        table.keep_only_columns(["C"])
