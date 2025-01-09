import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "max_missing_value_ratio", "expected"),
    [
        (
            Table({}),
            0,
            Table({}),
        ),
        (
            Table({"col1": []}),
            0,
            Table({}),  # All values in the column are missing
        ),
        (
            Table({"col1": [1, 2], "col2": [3, 4]}),
            0,
            Table({"col1": [1, 2], "col2": [3, 4]}),
        ),
        (
            Table({"col1": [1, 2, 3], "col2": [1, 2, None], "col3": [1, None, None]}),
            0,
            Table({"col1": [1, 2, 3]}),
        ),
        (
            Table({"col1": [1, 2, 3], "col2": [1, 2, None], "col3": [1, None, None]}),
            0.5,
            Table({"col1": [1, 2, 3], "col2": [1, 2, None]}),
        ),
        (
            Table({"col1": [1, 2, 3], "col2": [1, 2, None], "col3": [1, None, None]}),
            1,
            Table({"col1": [1, 2, 3], "col2": [1, 2, None], "col3": [1, None, None]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "no missing values",
        "some missing values (max_missing_value_ratio=0)",
        "some missing values (max_missing_value_ratio=0.5)",
        "some missing values (max_missing_value_ratio=1)",
    ],
)
def test_should_remove_columns_with_missing_values(table: Table, max_missing_value_ratio: int, expected: Table) -> None:
    updated_table = table.remove_columns_with_missing_values(max_missing_value_ratio=max_missing_value_ratio)
    assert updated_table == expected
