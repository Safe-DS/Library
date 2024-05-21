import re

import pytest
from safeds.data.labeled.containers import TimeSeriesDataset


@pytest.mark.parametrize(
    "tabular_dataset",
    [
        TimeSeriesDataset({"a": [1, 2], "b": [3, 4]}, target_name="b", time_name="a", window_size=1),
    ],
    ids=[
        "non-empty",
    ],
)
def test_should_contain_tabular_dataset_element(tabular_dataset: TimeSeriesDataset) -> None:
    pattern = r"<table.*?>.*?</table>"
    assert re.search(pattern, tabular_dataset._repr_html_(), flags=re.S) is not None


@pytest.mark.parametrize(
    "tabular_dataset",
    [
        TimeSeriesDataset({"a": [1, 2], "b": [3, 4]}, target_name="b", time_name="a", window_size=1),
    ],
    ids=[
        "non-empty",
    ],
)
def test_should_contain_th_element_for_each_column_name(tabular_dataset: TimeSeriesDataset) -> None:
    for column_name in tabular_dataset._table.column_names:
        assert f"<th>{column_name}</th>" in tabular_dataset._repr_html_()


@pytest.mark.parametrize(
    "tabular_dataset",
    [
        TimeSeriesDataset({"a": [1, 2], "b": [3, 4]}, target_name="b", time_name="a", window_size=1),
    ],
    ids=[
        "non-empty",
    ],
)
def test_should_contain_td_element_for_each_value(tabular_dataset: TimeSeriesDataset) -> None:
    for column in tabular_dataset._table.to_columns():
        for value in column:
            assert f"<td>{value}</td>" in tabular_dataset._repr_html_()
