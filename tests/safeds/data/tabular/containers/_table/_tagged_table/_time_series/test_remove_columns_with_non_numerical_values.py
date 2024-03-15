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
                    "feature_numerical": [0, 1, 2],
                    "feature_non_numerical": ["a", "b", "c"],
                    "non_feature_numerical": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_numerical", "feature_non_numerical"],
            ),
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_numerical": [0, 1, 2],
                    "non_feature_numerical": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_numerical"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_numerical": [0, 1, 2],
                    "non_feature_numerical": [7, 8, 9],
                    "non_feature_non_numerical": ["a", "b", "c"],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_numerical"],
            ),
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_numerical": [0, 1, 2],
                    "non_feature_numerical": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_numerical"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_numerical": [0, 1, 2],
                    "non_feature_numerical": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_numerical"],
            ),
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_numerical": [0, 1, 2],
                    "non_feature_numerical": [7, 8, 9],
                    "target": [3, 4, 5],
                },
                "target",
                "time",
                ["feature_numerical"],
            ),
        ),
    ],
    ids=["non_numerical_feature", "non_numerical_non_feature", "all_numerical"],
)
def test_should_remove_columns_with_non_numerical_values(table: TimeSeries, expected: TimeSeries) -> None:
    new_table = table.remove_columns_with_non_numerical_values()
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
                    "target": ["a", "b", "c"],
                },
                "target",
                "time",
                ["feature"],
            ),
            ColumnIsTargetError,
            r'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, "x", 2],
                    "non_feature": [1, 2, 3],
                    "target": ["a", "b", "c"],
                },
                "target",
                "time",
                ["feature"],
            ),
            ColumnIsTargetError,
            r'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, 1, 2],
                    "non_feature": [1, "x", 3],
                    "target": ["a", "b", "c"],
                },
                "target",
                "time",
                ["feature"],
            ),
            ColumnIsTargetError,
            r'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
        (
            TimeSeries(
                {
                    "time": ["!", "x", "2"],
                    "feature": [0, 1, 2],
                    "non_feature": [1, "x", 3],
                    "target": [1, 2, 3],
                },
                "target",
                "time",
                ["feature"],
            ),
            ColumnIsTimeError,
            r'Illegal schema modification: Column "time" is the time column and cannot be removed.',
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature": [0, "x", 2],
                    "non_feature": [1, "x", 3],
                    "target": ["a", "b", "c"],
                },
                "target",
                "time",
                ["feature"],
            ),
            ColumnIsTargetError,
            r'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
    ],
    ids=[
        "only_target_non_numerical",
        "also_feature_non_numerical",
        "also_non_feature_non_numerical",
        "time_non_numerical",
        "all_non_numerical",
    ],
)
def test_should_raise_in_remove_columns_with_non_numerical_values(
    table: TimeSeries,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error, match=error_msg):
        table.remove_columns_with_non_numerical_values()
