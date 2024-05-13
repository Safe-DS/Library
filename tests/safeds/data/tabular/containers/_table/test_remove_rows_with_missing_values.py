import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table(
                {
                    "col1": [None, None, "C", "A"],
                    "col2": [None, "2", "3", "4"],
                },
            ),
            Table(
                {
                    "col1": ["C", "A"],
                    "col2": ["3", "4"],
                },
            ),
        ),
        (Table(), Table()),
    ],
    ids=["some missing values", "empty"],
)
def test_should_remove_rows_with_missing_values(table: Table, expected: Table) -> None:
    updated_table = table.remove_rows_with_missing_values()
    assert updated_table.schema == expected.schema
    assert updated_table.number_of_columns == expected.number_of_columns
    assert updated_table == expected
