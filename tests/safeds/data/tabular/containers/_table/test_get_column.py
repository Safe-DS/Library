import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table1", "expected"),
    [
        (Table.from_dict({"col1": ["col1_1"], "col2": ["col2_1"]}),
         Column("col1", ["col1_1"])),
    ],
    ids=["First column"],
)
def test_should_get_column(table1: Table, expected: Column) -> None:
    assert table1.get_column("col1") == expected



def test_should_raise_error_if_column_name_unknown() -> None:
    table = Table.from_dict({"col1": ["col1_1"], "col2": ["col2_1"]})
    with pytest.raises(UnknownColumnNameError):
        table.get_column("col3")
