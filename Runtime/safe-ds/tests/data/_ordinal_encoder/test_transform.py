import pandas as pd
import pytest
from safeds.data import OrdinalEncoder
from safeds.data.tabular import Table
from safeds.exceptions import NotFittedError


def test_transform_invalid() -> None:
    test_table = Table(
        pd.DataFrame(
            {
                "temperatur": ["warm", "kalt", "kalt", "warm", "heiss"],
                "gedöns": ["1", "2", "3", "4", "5"],
                "temperatur_2": ["kalt", "kalt", "warm", "warm", "kalt"],
            }
        )
    )
    ode = OrdinalEncoder(["kalt", "warm", "heiss"])
    with pytest.raises(NotFittedError):
        ode.transform(test_table, "temperatur")
