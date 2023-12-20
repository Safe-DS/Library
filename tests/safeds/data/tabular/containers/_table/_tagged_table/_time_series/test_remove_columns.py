import pytest
from safeds.data.tabular.containers import Table, TimeSeries
from safeds.exceptions import ColumnIsTargetError, IllegalSchemaModificationError

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("table", "columns", "expected"),
    [
        (
            TimeSeries._from_table_to_time_series(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat_1": [1, 2, 3],
                        "feat_2": [4, 5, 6],
                        "non_feat_1": [2, 4, 6],
                        "non_feat_2": [3, 6, 9],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
                ["feat_1", "feat_2"],
            ),
            ["feat_2"],
            TimeSeries._from_table_to_time_series(
                Table({"time": [0, 1, 2], "feat_1": [1, 2, 3], "non_feat_1": [2, 4, 6], "non_feat_2": [3, 6, 9], "target": [7, 8, 9]}),
                "target",
                "time",
                ["feat_1"],
            ),
        ),
        (
            TimeSeries._from_table_to_time_series(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat_1": [1, 2, 3],
                        "feat_2": [4, 5, 6],
                        "non_feat_1": [2, 4, 6],
                        "non_feat_2": [3, 6, 9],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
                ["feat_1", "feat_2"],
            ),
            ["non_feat_2"],
            TimeSeries._from_table_to_time_series(
                Table({"time": [0, 1, 2],"feat_1": [1, 2, 3], "feat_2": [4, 5, 6], "non_feat_1": [2, 4, 6], "target": [7, 8, 9]}),
                "target",
                "time",
                ["feat_1", "feat_2"],

            ),
        ),
        (
            TimeSeries._from_table_to_time_series(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat_1": [1, 2, 3],
                        "feat_2": [4, 5, 6],
                        "non_feat_1": [2, 4, 6],
                        "non_feat_2": [3, 6, 9],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
                ["feat_1", "feat_2"],
            ),
            ["non_feat_1", "non_feat_2"],
            TimeSeries._from_table_to_time_series(
                Table({"time": [0, 1, 2],"feat_1": [1, 2, 3], "feat_2": [4, 5, 6], "target": [7, 8, 9]}),
                "target",
                "time",
                ["feat_1", "feat_2"],
            ),
        ),
        (
            TimeSeries._from_table_to_time_series(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat_1": [1, 2, 3],
                        "feat_2": [4, 5, 6],
                        "non_feat_1": [2, 4, 6],
                        "non_feat_2": [3, 6, 9],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
                ["feat_1", "feat_2"],
            ),
            ["feat_2", "non_feat_2"],
            TimeSeries._from_table_to_time_series(
                Table({"time": [0, 1, 2], "feat_1": [1, 2, 3], "non_feat_1": [2, 4, 6], "target": [7, 8, 9]}),
                "target",
                "time",
                ["feat_1"],
            ),
        ),
        (
            TimeSeries._from_table_to_time_series(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat_1": [1, 2, 3],
                        "non_feat_1": [2, 4, 6],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
                ["feat_1"],
            ),
            [],
            TimeSeries._from_table_to_time_series(
                Table({"time": [0, 1, 2],"feat_1": [1, 2, 3], "non_feat_1": [2, 4, 6], "target": [7, 8, 9]}),
                "target",
                "time",
                ["feat_1"],
            ),
        ),
    ],
    ids=[
        "remove_feature",
        "remove_non_feature",
        "remove_all_non_features",
        "remove_some_feat_and_some_non_feat",
        "remove_nothing",
    ],
)
def test_should_remove_columns(table: TimeSeries, columns: list[str], expected: TimeSeries) -> None:
    new_table = table.remove_columns(columns)
    assert_that_time_series_are_equal(new_table, expected)


@pytest.mark.parametrize(
    ("table", "columns", "error", "error_msg"),
    [
        (
            TimeSeries._from_table_to_time_series(
                Table({"time": [0, 1, 2],"feat": [1, 2, 3], "non_feat": [1, 2, 3], "target": [4, 5, 6]}),
                "target",
                "time",
                ["feat"],
            ),
            ["target"],
            ColumnIsTargetError,
            r'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
        (
            TimeSeries._from_table_to_time_series(
                Table({"time": [0, 1, 2],"feat": [1, 2, 3], "non_feat": [1, 2, 3], "target": [4, 5, 6]}),
                "target",
                "time",
                ["feat"],
            ),
            ["non_feat", "target"],
            ColumnIsTargetError,
            r'Illegal schema modification: Column "target" is the target column and cannot be removed.',
        ),
        (
            TimeSeries._from_table_to_time_series(
                Table({"time": [0, 1, 2], "feat": [1, 2, 3], "non_feat": [1, 2, 3], "target": [4, 5, 6]}),
                "target",
                "time",
                ["feat"],
            ),
            ["feat"],
            IllegalSchemaModificationError,
            r"Illegal schema modification: You cannot remove every feature column.",
        ),
        (
            TimeSeries._from_table_to_time_series(
                Table({"time": [0, 1, 2], "feat": [1, 2, 3], "non_feat": [1, 2, 3], "target": [4, 5, 6]}),
                "target",
                "time",
                ["feat"],
            ),
            ["feat", "non_feat"],
            IllegalSchemaModificationError,
            r"Illegal schema modification: You cannot remove every feature column.",
        ),
    ],
    ids=[
        "remove_only_target",
        "remove_non_feat_and_target",
        "remove_all_features",
        "remove_non_feat_and_all_features",
    ],
)
def test_should_raise_in_remove_columns(
    table: TimeSeries,
    columns: list[str],
    error: type[Exception],
    error_msg: str,
) -> None:
    with pytest.raises(error, match=error_msg):
        table.remove_columns(columns)
