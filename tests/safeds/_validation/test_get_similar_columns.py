import pytest

from safeds._validation._check_columns_exist_module import _get_similar_column_names
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table", "name", "expected"),
    [
        (
            Table({}),
            "column1",
            [],
        ),
        (
            Table({"column1": [], "column2": []}),
            "column1",
            ["column1"],
        ),
        (
            Table({"column1": [], "column2": [], "column3": []}),
            "dissimilar",
            [],
        ),
        (
            Table({"column1": [], "x": [], "y": []}),
            "cilumn1",
            ["column1"],
        ),
        (
            Table({"column1": [], "column2": [], "y": []}),
            "cilumn1",
            ["column1", "column2"],
        ),
    ],
    ids=["empty table", "exact match", "no similar", "one similar", "multiple similar"],
)
def test_should_get_similar_column_names(table: Table, name: str, expected: list[str]) -> None:
    assert _get_similar_column_names(table.schema, name) == expected
