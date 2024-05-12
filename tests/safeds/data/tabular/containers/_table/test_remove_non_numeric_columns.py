import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table(
                {
                    "col1": ["text", "text", "word", "word"],
                    "col3": [2, 3, 1, 4],
                },
            ),
            Table(
                {
                    "col3": [2, 3, 1, 4],
                },
            ),
        ),
        (Table(), Table()),
    ],
    ids=["numerical values", "empty"],
)
def test_should_remove_non_numeric_columns(table: Table, expected: Table) -> None:
    updated_table = table.remove_non_numeric_columns()
    assert updated_table.schema == expected.schema
    assert updated_table == expected
