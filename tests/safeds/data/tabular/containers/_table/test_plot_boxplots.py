import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import NonNumericColumnError


@pytest.mark.parametrize(
    "table",
    [
        Table({"A": [1, 2, 3]}),
        Table({"A": [1, 2, 3], "B": ["A", "A", "Bla"], "C": [True, True, False], "D": [1.0, 2.1, 4.5]}),
        Table({"A": [1, 2, 3], "B": [1.0, 2.1, 4.5], "C": [1, 2, 3], "D": [1.0, 2.1, 4.5]}),
    ],
    ids=["one column", "four columns (some non-numeric)", "four columns (all numeric)"],
)
def test_should_match_snapshot(table: Table, snapshot_png) -> None:
    boxplots = table.plot_boxplots()
    assert boxplots == snapshot_png


def test_should_raise_if_column_contains_non_numerical_values() -> None:
    table = Table.from_dict({"A": ["1", "2", "3.5"], "B": ["0.2", "4", "77"]})
    with pytest.raises(
        NonNumericColumnError,
        match=(
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThis table contains only"
            r" non-numerical columns."
        ),
    ):
        table.plot_boxplots()


def test_should_fail_on_empty_table() -> None:
    with pytest.raises(NonNumericColumnError):
        Table().plot_boxplots()
