import pytest
from safeds.data.tabular.containers import Table, TimeSeries
from safeds.exceptions import UnknownColumnNameError
from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("data", "time_name", "target_name", "feature_names", "error", "error_msg"),
    [
        (
            {
                "time": [0, 1],
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "time",
            "T",
            ["A", "B", "C", "D", "E"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D, E'",
        ),
        (
            {
                "time": [0, 1],
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "time",
            "D",
            ["A", "B", "C"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D'",
        ),
        (
            {
                "time": [0, 1],
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "time",
            "A",
            ["A", "B", "C"],
            ValueError,
            r"Column 'A' can not be time and feature column.",
        ),
        (
            {
                "time": [0, 1],
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "random",
            "B",
            ["A"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'random'.",
        ),
        (
            {
                "time": [0, 1],
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "time",
            "T",
            ["A", "B", "C", "time"],
            ValueError,
            "Column 'time' can not be time and feature column.",
        ),
    ],
    ids=[
        "feature_does_not_exist",
        "target_does_not_exist",
        "target_and_feature_overlap",
        "time_column_does_not_exist",
        "time_is_also_feature",
    ],
)
def test_should_raise_error(
    data: dict[str, list[int]],
    time_name: str,
    target_name: str,
    feature_names: list[str] | None,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error, match=error_msg):
        TimeSeries(data, target_name=target_name, time_name=time_name, feature_names=feature_names)


@pytest.mark.parametrize(
    ("data", "time_name", "target_name", "feature_names"),
    [
        (
            {
                "time": [0, 1],
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "time",
            "T",
            ["A", "B", "C"],
        ),
        (
            {
                "time": [0, 1],
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "time",
            "T",
            ["A", "C"],
        ),
        (
            {
                "time": [0, 1],
                "A": [1, 4],
                "B": [2, 5],
                "C": [3, 6],
                "T": [0, 1],
            },
            "time",
            "T",
            None,
        ),
    ],
    ids=["create_tagged_table", "tagged_table_not_all_columns_are_features", "tagged_table_with_feature_names_as_None"],
)
def test_should_create_a_time_series(
    data: dict[str, list[int]],
    time_name: str,
    target_name: str,
    feature_names: list[str] | None,
) -> None:
    time_series = TimeSeries(data, target_name=target_name, time_name=time_name, feature_names=feature_names)
    if feature_names is None:
        feature_names = []

    assert isinstance(time_series, TimeSeries)
    assert time_series._feature_names == feature_names
    assert time_series._target.name == target_name
    assert time_series._features == Table(data).keep_only_columns(feature_names)
    assert time_series._target == Table(data).get_column(target_name)
    assert time_series.time == Table(data).get_column(time_name)
