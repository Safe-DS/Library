import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [Table(), Table({"a": [], "b": []}), Table({"a": ["a", 3, 0.1], "b": [True, False, None]})],
    ids=["empty", "empty-rows", "normal"],
)
def test_should_copy_table(table: Table) -> None:
    copied = table._copy()
    assert copied == table
    assert copied is not table
