# pylint: disable=pointless-statement

import pandas as pd
import pytest
from safeds.data import Table
from safeds.exceptions import UnknownColumnNameError


# noinspection PyStatementEffect
def test_getitem_invalid() -> None:
    with pytest.raises(UnknownColumnNameError):
        table = Table(pd.DataFrame(data={"col1": [1], "col2": [2]}))
        row = table.get_row(0)
        row["col3"]


def test_getitem_valid() -> None:
    table = Table(pd.DataFrame(data={"col1": [1], "col2": [2]}))
    row = table.get_row(0)
    assert row["col1"] == 1 and row["col2"] == 2
