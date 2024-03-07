import sys

import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        Table(),
        Table({"col1": [0]}),
        Table({"col1": [0, "1"], "col2": ["a", "b"]}),
    ],
    ids=[
        "empty table",
        "table with one row",
        "table with multiple rows",
    ],
)
def test_should_size_be_greater_than_normal_object(table: Table) -> None:
    assert sys.getsizeof(table) > sys.getsizeof(object())
