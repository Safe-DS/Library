import pytest

from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table({}), Table({})),
        (Table({"col1": []}), Table({"col1": []})),
        (
            Table(
                {
                    "col1": [0, 1, 2, 1, 3],
                    "col2": [0, -1, -2, -1, -3],
                },
            ),
            Table({"col1": [0, 1, 2, 3], "col2": [0, -1, -2, -3]}),
        ),
    ],
    ids=["empty", "no rows", "duplicate rows"],
)
def test_should_remove_duplicate_rows(table: Table, expected: Table) -> None:
    actual = table.remove_duplicate_rows()
    assert actual == expected
