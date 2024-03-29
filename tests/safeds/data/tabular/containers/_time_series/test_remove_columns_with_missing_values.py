import pytest
from safeds.data.tabular.containers import TimeSeries
from safeds.exceptions import ColumnIsTargetError, ColumnIsTimeError

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_complete": [0, 1, 2],
                    "feature_incomplete": [3, None, 5],
                    "non_feature_complete": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_complete", "feature_incomplete"],
            ),
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_complete": [0, 1, 2],
                    "non_feature_complete": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_complete"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_complete": [0, 1, 2],
                    "non_feature_complete": [7, 8, 9],
                    "non_feature_incomplete": [3, None, 5],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_complete"],
            ),
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_complete": [0, 1, 2],
                    "non_feature_complete": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_complete"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_complete": [0, 1, 2],
                    "non_feature_complete": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_complete"],
            ),
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_complete": [0, 1, 2],
                    "non_feature_complete": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_complete"],
            ),
        ),
    ],
    ids=["incomplete_feature", "incomplete_non_feature", "all_complete"],
)
def test_should_remove_columns_with_non_numerical_values(table: TimeSeries, expected: TimeSeries) -> None:
    new_table = table.remove_columns_with_missing_values()
    assert_that_time_series_are_equal(new_table, expected)


@pytest.mark.parametrize(
    ("table", "error", "error_msg"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, 1, 2],
                    "non_feature": [1, 2, 3],
                    "target": [3, None, 5],
                },
                "target",
                "time",
                ["feature"],
            ),
            ColumnIsTargetError,
            'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, None, 2],
                    "non_feature": [1, 2, 3],
                    "target": [None, 4, 5],
                },
                "target",
                "time",
                ["feature"],
            ),
            ColumnIsTargetError,
            'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
        (
            TimeSeries(
                {
                    "time": [0, None, 2],
                    "feature": [0, 1, 2],
                    "non_feature": [1, 2, 3],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature"],
            ),
            ColumnIsTimeError,
            'Illegal schema modification: Column "time" is the time column and cannot be removed.',
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, 1, 2],
                    "non_feature": [1, 2, 3],
                    "target": [3, 4, None],
                },
                "target",
                "time",
                ["feature"],
            ),
            ColumnIsTargetError,
            'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, None, 2],
                    "non_feature": [1, None, 3],
                    "target": [3, None, 5],
                },
                "target",
                "time",
                ["feature"],
            ),
            ColumnIsTargetError,
            'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
    ],
    ids=[
        "only_target_incomplete",
        "also_feature_incomplete",
        "time_is_incomplete",
        "also_non_feature_incomplete",
        "all_incomplete",
    ],
)
def test_should_raise_in_remove_columns_with_missing_values(
    table: TimeSeries,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(
        error,
        match=error_msg,
    ):
        table.remove_columns_with_missing_values()
