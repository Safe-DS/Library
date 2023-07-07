import pytest
from safeds.data.tabular.containers import Table

from safeds.exceptions._data import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table", "column_name", "expected"),
    [
        (Table({"column1": ["col1_1"], "col2": ["col2_1"], "cilumn2": ["cil2_1"]}), "col1", ["column1", "col2"]),
        (Table({"column1": ["col1_1"], "col2": ["col2_1"], "cilumn2": ["cil2_1"]}), "clumn1", ["column1", "cilumn2"]),
    ],
    ids=["one similar", "two similar"],
)
def test_should_get_similar_column_names(table: Table, column_name: str, expected: list[str]) -> None:
    assert table._get_similar_columns(column_name) == expected


def test_should_raise_error_if_column_name_unknown() -> None:
    with pytest.raises(UnknownColumnNameError, match=r"Could not find column\(s\) 'col3'.\nDid you mean '\['col1', 'col2'\]'?"):
        raise UnknownColumnNameError(["col3"], ["col1", "col2"])
