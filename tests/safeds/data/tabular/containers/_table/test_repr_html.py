import re

import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "table",
    [
        Table({}),
        Table({"col1": []}),
        Table({"col1": [1, 2], "col2": [3, 4]}),
    ],
    ids=[
        "empty",
        "no rows",
        "with data",
    ],
)
class TestHtml:
    def test_should_contain_table_element(self, table: Table) -> None:
        actual = table._repr_html_()
        pattern = r"<table.*?>.*?</table>"
        assert re.search(pattern, actual, flags=re.S) is not None

    def test_should_contain_th_element_for_each_column_name(self, table: Table) -> None:
        actual = table._repr_html_()
        for column_name in table.column_names:
            assert f"<th>{column_name}</th>" in actual

    def test_should_contain_td_element_for_each_value(self, table: Table) -> None:
        actual = table._repr_html_()
        for column in table.to_columns():
            for value in column:
                assert f"<td>{value}</td>" in actual
