import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError

@pytest.mark.parametrize(
    ("table", "table_transformed"),
    [
        (Table.from_dict({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["a", "b", "c"]}),
         Table.from_dict({"A": [2, 4, 6], "B": [4, 5, 6], "C": ["a", "b", "c"]}))
    ],
    ids=["multiply by 2"]
)
def test_should_transform_column(table: Table, table_transformed: Table) -> None:
    result = table.transform_column("A", lambda row: row.get_value("A") * 2)

    assert result == table_transformed


def test_should_raise_if_column_not_found() -> None:
    input_table = Table.from_dict(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": ["a", "b", "c"],
        },
    )

    with pytest.raises(UnknownColumnNameError):
        input_table.transform_column("D", lambda row: row.get_value("A") * 2)
