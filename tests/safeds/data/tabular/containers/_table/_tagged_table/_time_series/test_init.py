import pytest
from safeds.data.tabular.containers import Table, TimeSeries
from safeds.exceptions import UnknownColumnNameError


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
            r"Column 'A' cannot be both feature and target.",
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
            [],
            ValueError,
            r"At least one feature column must be specified.",
        ),
        (
            {
                "time": [0, 1],
                "A": [1, 4],
            },
            "time",
            "A",
            None,
            ValueError,
            r"At least one feature column must be specified.",
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
            ValueError,
            r"Column 'random' must exist in the table.",
        ),
    ],
    ids=[
        "feature_does_not_exist",
        "target_does_not_exist",
        "target_and_feature_overlap",
        "features_are_empty-explicitly",
        "features_are_empty_implicitly",
        "time_column_does_not_exist",
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
def test_should_create_a_tagged_table(
    data: dict[str, list[int]],
    time_name: str,
    target_name: str,
    feature_names: list[str] | None,
) -> None:
    time_series = TimeSeries(data, target_name=target_name, time_name=time_name, feature_names=feature_names)
    if feature_names is None:
        feature_names = list(data.keys())
        feature_names.remove(target_name)
        feature_names.remove(time_name)
    assert isinstance(time_series, TimeSeries)
    assert time_series._features.column_names == feature_names
    assert time_series._target.name == target_name
    assert time_series._features == Table(data).keep_only_columns(feature_names)
    assert time_series._target == Table(data).get_column(target_name)
    assert time_series.time == Table(data).get_column(time_name)
