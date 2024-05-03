import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import NonNumericColumnError
from syrupy import SnapshotAssertion


def create_time_series_list() -> list[Column]:
    table1 = Column(
            "target", [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    )
    table2 = Column(
            "target", [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    )
    return [table1, table2]


def create_invalid_time_series_list() -> list[Column]:
    table1 = Column(
            "target", ["9", 10, 11, 12, 13, 14, 15, 16, 17, 18]
    )
    table2 = Column(

            "target", ["4", 5, 6, 7, 8, 9, 10, 11, 12, 13]

    )
    return [table1, table2]


def test_legit_compare(snapshot_png_image: SnapshotAssertion) -> None:
    col = Column(

            "target", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    )
    plot = col.plot_compare_columns(create_time_series_list())
    assert plot == snapshot_png_image


def test_should_raise_if_column_contains_non_numerical_values_x() -> None:
    table = Column(

            "target", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    )
    with pytest.raises(
        NonNumericColumnError,
        match=(
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThe time series plotted"
            r" column"
            r" contains"
            r" non-numerical columns."
        ),
    ):
        table.plot_compare_columns(create_time_series_list())


def test_with_non_valid_list() -> None:
    table = Column(

            "target", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    )
    with pytest.raises(
        NonNumericColumnError,
        match=(
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \nThe time series plotted"
            r" column"
            r" contains"
            r" non-numerical columns."
        ),
    ):
        table.plot_compare_columns(create_invalid_time_series_list())

def test_with_non_valid_list() -> None:
    table = Column(

            "target", [1, 2, 3, 4, 5, 6, 7, 8,],

    )
    with pytest.raises(
        ValueError,
        match=(
            r"The columns must have the same size."
        ),
    ):
        table.plot_compare_columns(create_time_series_list())
