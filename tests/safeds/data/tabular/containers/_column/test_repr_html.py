import pytest
import regex as re
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    "column",
    [
        Column("a", []),
        Column("a", [1, 2, 3]),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_contain_table_element(column: Column) -> None:
    pattern = r"<table.*?>.*?</table>"
    assert re.search(pattern, column._repr_html_(), flags=re.S) is not None


@pytest.mark.parametrize(
    "column",
    [
        Column("a", []),
        Column("a", [1, 2, 3]),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_contain_th_element_for_column_name(column: Column) -> None:
    assert f"<th>{column.name}</th>" in column._repr_html_()


@pytest.mark.parametrize(
    "column",
    [
        Column("a", []),
        Column("a", [1, 2, 3]),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_contain_td_element_for_each_value(column: Column) -> None:
    for value in column:
        assert f"<td>{value}</td>" in column._repr_html_()
