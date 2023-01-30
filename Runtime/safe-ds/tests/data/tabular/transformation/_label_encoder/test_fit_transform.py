import pandas as pd
from safeds.data.tabular import Table
from safeds.data.tabular.transformation import LabelEncoder


def test_fit_transform_valid() -> None:
    test_table = Table(
        pd.DataFrame({"cities": ["paris", "paris", "tokyo", "amsterdam"]})
    )
    le = LabelEncoder()
    test_table = le.fit_transform(test_table, ["cities"])
    assert test_table.schema.has_column("cities")
    assert test_table.to_columns()[0].get_value(0) == 1
    assert test_table.to_columns()[0].get_value(2) == 2
    assert test_table.to_columns()[0].get_value(3) == 0
