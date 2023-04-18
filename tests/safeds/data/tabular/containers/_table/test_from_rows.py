import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.exceptions import MissingDataError


def test_from_rows() -> None:
    table_expected = Table.from_dict(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "D": ["d", "e", "f"],
        }
    )
    rows_is = table_expected.to_rows()
    table_is = Table.from_rows(rows_is)

    assert table_is == table_expected


def test_from_rows_invalid() -> None:
    with pytest.raises(MissingDataError):
        Table.from_rows([])
