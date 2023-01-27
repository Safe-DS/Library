import pandas as pd
import pytest
from safe_ds.data import OrdinalEncoder, Table
from safe_ds.exceptions import NotFittedError


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
