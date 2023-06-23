import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table1", "column_name", "expected"),
    [
        (
            Table({"column1": ["col1_1"], "col2": ["col2_1"], "cilumn2": ["cil2_1"]}),
            "col1",
            ['col2']
        ),
        (
            Table({"column1": ["col1_1"], "col2": ["col2_1"], "cilumn2": ["cil2_1"]}),
            "clumn1",
            ['column1', 'column3']
        )
    ],
    ids=["one similar", "two similar"]
)
def test_should_warn_if_similar_column_name(table1: Table, column_name: str, expected: list[str]) -> None:
    with pytest.warns(
        UserWarning,
        match=(
            f"did you mean one of these: {expected}?"
        ),
    ):
        table1.get_similar_columns(column_name)
