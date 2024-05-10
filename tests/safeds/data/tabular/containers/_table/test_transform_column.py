import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table", "table_transformed"),
    [
        (
            Table({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["a", "b", "c"]}),
            Table({"A": [2, 4, 6], "B": [4, 5, 6], "C": ["a", "b", "c"]}),
        ),
    ],
    ids=["multiply by 2"],
)
def test_should_transform_column(table: Table_transformed: Table) -> None:
    result = table.transform_column("A", lambda row: row.get_value("A") * 2)

    assert result.schema == table_transformed.schema
    assert result == table_transformed


@pytest.mark.parametrize(
    "table",
    [
        Table(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": ["a", "b", "c"],
            },
        ),
        Table(),
    ],
    ids=["column not found", "empty"],
)
def test_should_raise_if_column_not_found(table: Table) -> None:
    with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'D'"):
        table.transform_column("D", lambda row: row.get_value("A") * 2)
