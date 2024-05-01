import re

import pytest
from safeds.data.labeled.containers import TabularDataset


@pytest.mark.parametrize(
    "tabular_dataset",
    [
        TabularDataset({"a": [1, 2], "b": [3, 4]}, target_name="b"),
    ],
    ids=[
        "non-empty",
    ],
)
def test_should_contain_tabular_dataset_element(tabular_dataset: TabularDataset) -> None:
    pattern = r"<table.*?>.*?</table>"
    assert re.search(pattern, tabular_dataset._repr_html_(), flags=re.S) is not None


@pytest.mark.parametrize(
    "tabular_dataset",
    [
        TabularDataset({"a": [1, 2], "b": [3, 4]}, target_name="b"),
    ],
    ids=[
        "non-empty",
    ],
)
def test_should_contain_th_element_for_each_column_name(tabular_dataset: TabularDataset) -> None:
    for column_name in tabular_dataset._table.column_names:
        assert f"<th>{column_name}</th>" in tabular_dataset._repr_html_()


@pytest.mark.parametrize(
    "tabular_dataset",
    [
        TabularDataset({"a": [1, 2], "b": [3, 4]}, target_name="b"),
    ],
    ids=[
        "non-empty",
    ],
)
def test_should_contain_td_element_for_each_value(tabular_dataset: TabularDataset) -> None:
    for column in tabular_dataset._table.to_columns():
        for value in column:
            assert f"<td>{value}</td>" in tabular_dataset._repr_html_()
