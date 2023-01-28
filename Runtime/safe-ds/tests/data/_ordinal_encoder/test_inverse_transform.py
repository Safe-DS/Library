import pandas as pd
from safeds.data import OrdinalEncoder
from safeds.data.tabular import Table
from safeds.data.tabular.typing import StringColumnType


def test_inverse_transform() -> None:
    test_table = Table(
        pd.DataFrame(
            {
                "temperatur": [1, 0, 0, 1, 2],
                "gedöns": ["1", "2", "3", "4", "5"],
                "temperatur_2": [0, 0, 1, 1, 0],
            }
        )
    )
    check_table = Table(
        pd.DataFrame(
            {
                "temperatur": ["warm", "kalt", "kalt", "warm", "heiss"],
                "gedöns": ["1", "2", "3", "4", "5"],
                "temperatur_2": ["kalt", "kalt", "warm", "warm", "kalt"],
            }
        )
    )
    oe = OrdinalEncoder(["kalt", "warm", "heiss"])
    oe.fit(check_table, "temperatur")
    test_table = oe.inverse_transform(test_table, "temperatur")
    test_table = oe.inverse_transform(test_table, "temperatur_2")
    assert test_table.schema.get_column_names() == check_table.schema.get_column_names()
    assert isinstance(
        test_table.schema.get_type_of_column("temperatur"), StringColumnType
    )
    assert isinstance(
        test_table.schema.get_type_of_column("temperatur_2"), StringColumnType
    )
    assert test_table == check_table
