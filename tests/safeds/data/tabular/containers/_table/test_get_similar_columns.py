import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions._data import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table", "column_name", "expected"),
    [
        (Table({"column1": ["col1_1"], "x": ["y"], "cilumn2": ["cil2_1"]}), "col1", ["column1"]),
        (
            Table(
                {
                    "column1": ["col1_1"],
                    "col2": ["col2_1"],
                    "col3": ["col2_1"],
                    "col4": ["col2_1"],
                    "cilumn2": ["cil2_1"],
                },
            ),
            "clumn1",
            ["column1", "cilumn2"],
        ),
        (
            Table({"column1": ["a"], "column2": ["b"], "column3": ["c"]}),
            "notexisting",
            [],
        ),
        (
            Table({"column1": ["col1_1"], "x": ["y"], "cilumn2": ["cil2_1"]}),
            "x",
            ["x"],
        ),
        (Table({}), "column1", []),
    ],
    ids=["one similar", "two similar/ dynamic increase", "no similar", "exact match", "empty table"],
)
def test_should_get_similar_column_names(table: Table, column_name: str, expected: list[str]) -> None:
    assert table._get_similar_columns(column_name) == expected


def test_should_raise_error_if_column_name_unknown() -> None:
    with pytest.raises(
        ColumnNotFoundError,
        match=r"Could not find column\(s\) 'col3'.\nDid you mean '\['col1', 'col2'\]'?",
    ):
        raise ColumnNotFoundError(["col3"], ["col1", "col2"])
