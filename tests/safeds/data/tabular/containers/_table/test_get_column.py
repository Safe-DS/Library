import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table1", "expected"),
    [
        (Table({"col1": ["col1_1"], "col2": ["col2_1"]}), Column("col1", ["col1_1"])),
    ],
    ids=["First column"],
)
def test_should_get_column(table1: Table, expected: Column) -> None:
    assert table1.get_column("col1") == expected


@pytest.mark.parametrize(
    "table",
    [
        (Table({"col1": ["col1_1"], "col2": ["col2_1"]})),
        (Table()),
    ],
    ids=["no col3", "empty"],
)
def test_should_raise_error_if_column_name_unknown(table: Table) -> None:
    with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'col3'"):
        table.get_column("col3")

def test_should_warn_if_similar_column_name() -> None:
    table1 = Table({"col1": ["col1_1"], "col2": ["col2_1"]})
    with pytest.warns(
        UserWarning,
        match=(
            f"did you mean col1?"
        ),
    ):
        table1.get_similar_columns("cil1")
