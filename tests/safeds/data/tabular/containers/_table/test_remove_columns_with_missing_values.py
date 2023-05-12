import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            (
                Table(
                    {
                        "col1": [None, None, None, None],
                        "col2": [1, 2, 3, None],
                        "col3": [1, 2, 3, 4],
                        "col4": [2, 3, 1, 4],
                    },
                ),
                Table(
                    {
                        "col3": [1, 2, 3, 4],
                        "col4": [2, 3, 1, 4],
                    }
                ),
            )
        ),
        (Table({}), Table({})),
    ],
    ids=["some missing values", "empty"],
)
def test_should_remove_columns_with_missing_values(table: Table, expected: Table) -> None:
    updated_table = table.remove_columns_with_missing_values()
    assert updated_table.schema == expected.schema
    assert updated_table.column_names == expected.column_names
