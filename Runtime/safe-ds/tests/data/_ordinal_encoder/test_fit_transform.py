import pandas as pd
import pytest
from safeds.data import OrdinalEncoder
from safeds.data.tabular import Table
from safeds.data.tabular.typing import IntColumnType
from safeds.exceptions import NotFittedError


def test_fit_transform_valid() -> None:
    test_table = Table(
        pd.DataFrame(
            {
                "temperatur": ["warm", "kalt", "kalt", "warm", "heiss"],
                "gedöns": ["1", "2", "3", "4", "5"],
                "temperatur_2": ["kalt", "kalt", "warm", "warm", "kalt"],
            }
        )
    )
    check_table = Table(
        pd.DataFrame(
            {
                "temperatur": [1, 0, 0, 1, 2],
                "gedöns": ["1", "2", "3", "4", "5"],
                "temperatur_2": [0, 0, 1, 1, 0],
            }
        )
    )
    oe = OrdinalEncoder(["kalt", "warm", "heiss"])
    test_table = oe.fit_transform(test_table, ["temperatur", "temperatur_2"])
    assert test_table.schema.get_column_names() == check_table.schema.get_column_names()
    assert isinstance(test_table.schema.get_type_of_column("temperatur"), IntColumnType)
    assert isinstance(
        test_table.schema.get_type_of_column("temperatur_2"), IntColumnType
    )
    assert test_table == check_table


def test_fit_transform_invalid() -> None:
    oe = OrdinalEncoder(["test", "test"])
    test_table = Table(
        pd.DataFrame(
            {
                "temperatur": ["warm", "kalt", "kalt", "warm", "heiss"],
                "gedöns": ["1", "2", "3", "4", "5"],
                "temperatur_2": ["kalt", "kalt", "warm", "warm", "kalt"],
            }
        )
    )
    with pytest.raises(NotFittedError):
        oe.transform(test_table, "test")
