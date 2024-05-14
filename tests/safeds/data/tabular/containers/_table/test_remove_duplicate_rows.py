import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            Table(
                {
                    "A": [1, 1, 1, 4],
                    "B": [2, 2, 2, 5],
                },
            ),
            Table({"A": [1, 4], "B": [2, 5]}),
        ),
        (Table(), Table()),
        (Table({"col1": []}), Table({"col1": []})),
    ],
    ids=["duplicate rows", "empty", "no rows"],
)
def test_should_remove_duplicate_rows(table: Table, expected: Table) -> None:
    result_table = table.remove_duplicate_rows()
    assert result_table.schema == expected.schema
    assert result_table == expected
