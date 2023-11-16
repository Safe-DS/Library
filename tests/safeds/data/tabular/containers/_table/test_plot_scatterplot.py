import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError


def test_should_match_snapshot(snapshot_png) -> None:
    table = Table({"A": [1, 2, 3], "B": [2, 4, 7]})
    scatterplot = table.plot_scatterplot("A", "B")
    assert scatterplot == snapshot_png


@pytest.mark.parametrize(
    ("table", "col1", "col2", "error_message"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "A", r"Could not find column\(s\) 'C'"),
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "B", "C", r"Could not find column\(s\) 'C'"),
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "D", r"Could not find column\(s\) 'C, D'"),
        (Table(), "C", "D", r"Could not find column\(s\) 'C, D'"),
    ],
    ids=["First argument doesn't exist", "Second argument doesn't exist", "Both arguments do not exist", "empty"],
)
def test_should_raise_if_column_does_not_exist(table: Table, col1: str, col2: str, error_message: str) -> None:
    with pytest.raises(UnknownColumnNameError, match=error_message):
        table.plot_scatterplot(col1, col2)
