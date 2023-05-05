import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         "   col1  col2\n0     1     1\n1     2     2\n2     1     4"),
    ],
    ids=["multiple rows"],
)
def test_should_return_a_string_representation(table: Table, expected: str) -> None:
    assert str(table) == expected
