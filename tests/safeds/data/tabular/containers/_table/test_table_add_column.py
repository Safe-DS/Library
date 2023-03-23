import pandas as pd
import pytest

from tests.fixtures import resolve_resource_path
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError


def test_table_add_column_valid() -> None:
    input_table = Table.from_csv(
        resolve_resource_path("test_table_add_column_valid_input.csv")
    )
    expected = Table.from_csv(resolve_resource_path("test_table_add_column_valid_output.csv"))
    column = Column(pd.Series(["a", "b", "c"]), "C")

    result = input_table.add_column(column)
    assert expected == result


@pytest.mark.parametrize(
    "column_values, column_name, error",
    [
        (["a", "b", "c"], "B", DuplicateColumnNameError),
        (["a", "b"], "C", ColumnSizeError),
    ],
)
def test_table_add_column_(
    column_values: list[str], column_name: str, error: type[Exception]
) -> None:
    input_table = Table.from_csv(
        resolve_resource_path("test_table_add_column_valid_input.csv")
    )
    column = Column(pd.Series(column_values), column_name)

    with pytest.raises(error):
        input_table.add_column(column)
