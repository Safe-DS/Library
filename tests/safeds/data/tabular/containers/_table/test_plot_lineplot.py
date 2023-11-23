import pytest
from safeds.data.image.containers import ImagePil
from safeds.data.tabular.containers import Table
from safeds.exceptions import UnknownColumnNameError

from tests.helpers import resolve_resource_path


def test_should_match_snapshot() -> None:
    table = Table({"A": [1, 2, 3], "B": [2, 4, 7]})
    current = table.plot_lineplot("A", "B")
    snapshot = ImagePil.from_png_file(resolve_resource_path("./image/snapshot_lineplot.png"))

    # Inlining the expression into the assert causes pytest to hang if the assertion fails when run from PyCharm.
    assertion = snapshot._image.tobytes() == current._image.tobytes()
    assert assertion


@pytest.mark.parametrize(
    ("table", "x", "y"),
    [
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "C", "A"),
        (Table({"A": [1, 2, 3], "B": [2, 4, 7]}), "A", "C"),
        (Table(), "x", "y"),
    ],
    ids=["x column", "y column", "empty"],
)
def test_should_raise_if_column_does_not_exist(table: Table, x: str, y: str) -> None:
    table = Table({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(UnknownColumnNameError):
        table.plot_lineplot(x, y)


@pytest.mark.parametrize(
    ("x", "y", "error_message"),
    [
        ("C", "A", r"Could not find column\(s\) 'C'"),
        ("A", "C", r"Could not find column\(s\) 'C'"),
        ("C", "D", r"Could not find column\(s\) 'C, D'"),
    ],
    ids=["x column", "y column", "x and y column"],
)
def test_should_raise_if_column_does_not_exist_error_message(x: str, y: str, error_message: str) -> None:
    table = Table({"A": [1, 2, 3], "B": [2, 4, 7]})
    with pytest.raises(UnknownColumnNameError, match=error_message):
        table.plot_lineplot(x, y)
