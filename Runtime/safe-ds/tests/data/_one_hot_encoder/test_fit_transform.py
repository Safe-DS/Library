import pandas as pd
from safeds.data import OneHotEncoder, Table


def test_fit_transform() -> None:
    table = Table(
        pd.DataFrame(
            data={
                "col1": ["A", "B", "C", "A"],
                "col2": ["Test1", "Test1", "Test3", "Test1"],
                "col3": [1, 2, 3, 4],
            }
        )
    )
    ohe = OneHotEncoder()
    table_ohe = ohe.fit_transform(table, ["col1", "col2"])
    assert table_ohe.count_columns() == 6
    assert table_ohe.get_row(0).get_value("col1_A") == 1
    assert table_ohe.get_row(1).get_value("col1_B") == 1
    assert table_ohe.get_row(2).get_value("col1_C") == 1
    assert table_ohe.get_row(3).get_value("col1_A") == 1
    assert table_ohe.get_row(0).get_value("col2_Test1") == 1
    assert table_ohe.get_row(1).get_value("col2_Test1") == 1
    assert table_ohe.get_row(2).get_value("col2_Test3") == 1
    assert table_ohe.get_row(3).get_value("col2_Test1") == 1
    assert table_ohe.get_column("col1_A").statistics.sum() == 2
    assert table_ohe.get_column("col1_B").statistics.sum() == 1
    assert table_ohe.get_column("col1_C").statistics.sum() == 1
    assert table_ohe.get_column("col2_Test1").statistics.sum() == 3
    assert table_ohe.get_column("col2_Test3").statistics.sum() == 1
    assert table_ohe.get_column("col3").statistics.sum() == 10
