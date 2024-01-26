import pytest
from safeds.data.tabular.containers import Table, TimeSeries
from safeds.exceptions import UnknownColumnNameError


@pytest.mark.parametrize(
    ("table", "target_name", "time_name", "feature_names", "error", "error_msg"),
    [
        (
            Table(
                {
                    "time": [0, 1],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            "time",
            ["A", "B", "C", "D", "E"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D, E'",
        ),
        (
            Table(
                {
                    "time": [0, 1],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "D",
            "time",
            ["A", "B", "C"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'D'",
        ),
        (
            Table(
                {
                    "time": [0, 1],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "A",
            "time",
            ["A", "B", "C"],
            ValueError,
            r"Column 'A' cannot be both feature and target.",
        ),
        (
            Table(
                {
                    "time": [0, 1],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "A",
            "time",
            [],
            ValueError,
            r"At least one feature column must be specified.",
        ),
        (
            Table(
                {
                    "time": [0, 1],
                    "A": [1, 4],
                },
            ),
            "A",
            "time",
            None,
            ValueError,
            r"At least one feature column must be specified.",
        ),
        (
            Table(
                {
                    "r": [0, 1],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            "time",
            ["A", "B", "C"],
            UnknownColumnNameError,
            r"Could not find column\(s\) 'time'",
        ),
        (
            Table(
                {
                    "time": [0, 1],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            "A",
            ["A", "B", "C"],
            ValueError,
            r"Column 'A' can not be time and feature column."
        ),
    ],
    ids=[
        "feature_does_not_exist",
        "target_does_not_exist",
        "target_and_feature_overlap",
        "features_are_empty-explicitly",
        "features_are_empty_implicitly",
        "time_does_not_exist",
        "time_is_also_feature",
    ],
)
def test_should_raise_error(
    table: Table,
    target_name: str,
    time_name: str,
    feature_names: list[str] | None,
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error, match=error_msg):
        TimeSeries._from_table_to_time_series(
            table,
            target_name=target_name,
            time_name=time_name,
            feature_names=feature_names,
        )


@pytest.mark.parametrize(
    ("table", "target_name", "time_name", "feature_names"),
    [
        (
            Table(
                {
                    "time": [0, 1],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            "time",
            ["A", "B", "C"],
        ),
        (
            Table(
                {
                    "time": [0, 1],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            "time",
            ["A", "C"],
        ),
        (
            Table(
                {
                    "time": [0, 1],
                    "A": [1, 4],
                    "B": [2, 5],
                    "C": [3, 6],
                    "T": [0, 1],
                },
            ),
            "T",
            "time",
            None,
        ),
    ],
    ids=["create_tagged_table", "tagged_table_not_all_columns_are_features", "tagged_table_with_feature_names_as_None"],
)
def test_should_create_a_tagged_table(
    table: Table,
    target_name: str,
    time_name: str,
    feature_names: list[str] | None,
) -> None:
    time_series = TimeSeries._from_table_to_time_series(
        table,
        target_name=target_name,
        time_name=time_name,
        feature_names=feature_names,
    )
    feature_names = (
        feature_names if feature_names is not None else table.remove_columns([target_name, time_name]).column_names
    )
    assert isinstance(time_series, TimeSeries)
    assert time_series._features.column_names == feature_names
    assert time_series._target.name == target_name
    assert time_series._features == table.keep_only_columns(feature_names)
    assert time_series._target == table.get_column(target_name)
    assert time_series.time == table.get_column(time_name)
