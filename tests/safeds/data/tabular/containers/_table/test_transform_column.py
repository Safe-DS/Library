import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import UnknownColumnNameError


def test_transform_column_valid() -> None:
    input_table = Table.from_dict(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": ["a", "b", "c"],
        }
    )

    result = input_table.transform_column("A", lambda row: row.get_value("A") * 2)

    assert result == Table.from_dict(
        {
            "A": [2, 4, 6],
            "B": [4, 5, 6],
            "C": ["a", "b", "c"],
        }
    )


def test_transform_column_invalid() -> None:
    input_table = Table.from_dict(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": ["a", "b", "c"],
        }
    )

    with pytest.raises(UnknownColumnNameError):
        input_table.transform_column("D", lambda row: row.get_value("A") * 2)
