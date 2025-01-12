import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import OutOfBoundsError


@pytest.mark.parametrize(
    ("table", "missing_value_ratio_threshold", "expected"),
    [
        (
            Table({}),
            0,
            Table({}),
        ),
        (
            Table({"col1": []}),
            0,
            Table({}),  # All values in the column are missing
        ),
        (
            Table({"col1": [1, 2], "col2": [3, 4]}),
            0,
            Table({"col1": [1, 2], "col2": [3, 4]}),
        ),
        (
            Table({"col1": [1, 2, 3], "col2": [1, 2, None], "col3": [1, None, None]}),
            0,
            Table({"col1": [1, 2, 3]}),
        ),
        (
            Table({"col1": [1, 2, 3], "col2": [1, 2, None], "col3": [1, None, None]}),
            0.5,
            Table({"col1": [1, 2, 3], "col2": [1, 2, None]}),
        ),
        (
            Table({"col1": [1, 2, 3], "col2": [1, 2, None], "col3": [1, None, None]}),
            1,
            Table({"col1": [1, 2, 3], "col2": [1, 2, None], "col3": [1, None, None]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "no missing values",
        "some missing values (missing_value_ratio_threshold=0)",
        "some missing values (missing_value_ratio_threshold=0.5)",
        "some missing values (missing_value_ratio_threshold=1)",
    ],
)
def test_should_remove_columns_with_missing_values(
    table: Table,
    missing_value_ratio_threshold: int,
    expected: Table,
) -> None:
    updated_table = table.remove_columns_with_missing_values(
        missing_value_ratio_threshold=missing_value_ratio_threshold,
    )
    assert updated_table == expected


@pytest.mark.parametrize(
    "missing_value_ratio_threshold",
    [
        -1,
        2,
    ],
    ids=[
        "too low",
        "too high",
    ],
)
def test_should_raise_if_missing_value_ratio_threshold_out_of_bounds(missing_value_ratio_threshold: float) -> None:
    with pytest.raises(OutOfBoundsError):
        Table({}).remove_columns_with_missing_values(missing_value_ratio_threshold=missing_value_ratio_threshold)
