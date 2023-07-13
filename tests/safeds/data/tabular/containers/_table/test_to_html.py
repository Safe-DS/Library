import re

import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        Table(),
        Table({"a": [1, 2], "b": [3, 4]}),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_contain_table_element(table: Table) -> None:
    pattern = r"<table.*?>.*?</table>"
    assert re.search(pattern, table.to_html(), flags=re.S) is not None


@pytest.mark.parametrize(
    "table",
    [
        Table(),
        Table({"a": [1, 2], "b": [3, 4]}),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_contain_th_element_for_each_column_name(table: Table) -> None:
    for column_name in table.column_names:
        assert f"<th>{column_name}</th>" in table.to_html()


@pytest.mark.parametrize(
    "table",
    [
        Table(),
        Table({"a": [1, 2], "b": [3, 4]}),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_contain_td_element_for_each_value(table: Table) -> None:
    for column in table.to_columns():
        for value in column:
            assert f"<td>{value}</td>" in table.to_html()
