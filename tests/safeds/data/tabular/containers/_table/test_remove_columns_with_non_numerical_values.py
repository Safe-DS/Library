import pytest

from safeds.data.tabular.containers import Table

@pytest.mark.parametrize(
    ("table", "expected"),
    [(Table.from_dict(
        {
            "col1": ["text", "text", "word", "word"],
            "col3": [2, 3, 1, 4],
        },
    ), ["col3"]), (Table([]), [])

    ],
    ids=["numerical values", "empty"],
)
def test_should_remove_columns_with_non_numerical_values(table: Table, expected: list) -> None:
    updated_table = table.remove_columns_with_non_numerical_values()
    assert updated_table.column_names == expected



