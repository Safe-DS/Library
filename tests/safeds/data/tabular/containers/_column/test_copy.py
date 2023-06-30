import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    "column",
    [Column("a"), Column("a", ["a", 3, 0.1])],
    ids=["empty", "normal"],
)
def test_should_copy_table(column: Column) -> None:
    copied = column._copy()
    assert copied == column
    assert copied is not column
