import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


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
def test_should_transform_column(table: Table, table_transformed: Table) -> None:
    result = table.transform_column("A", lambda cell: cell * 2)

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
    with pytest.raises(ColumnNotFoundError, match=r"Could not find column\(s\) 'D'"):
        table.transform_column("D", lambda cell: cell * 2)
