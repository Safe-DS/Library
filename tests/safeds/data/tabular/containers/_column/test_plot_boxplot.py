import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import NonNumericColumnError


def test_should_match_snapshot(snapshot_png) -> None:
    table = Table({"A": [1, 2, 3]})
    boxplot = table.get_column("A").plot_boxplot()
    assert boxplot == snapshot_png


def test_should_raise_if_column_contains_non_numerical_values() -> None:
    table = Table({"A": [1, 2, "A"]})
    with pytest.raises(NonNumericColumnError):
        table.get_column("A").plot_boxplot()
