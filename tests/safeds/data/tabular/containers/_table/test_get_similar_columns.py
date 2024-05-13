import pytest
from safeds._validation._check_columns_exist import _get_similar_column_names
from safeds.data.tabular.containers import Table


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
    assert _get_similar_column_names(table.schema, column_name) == expected  # TODO: move to validation tests
