import pytest
from safeds.data.tabular.containers import Table, TimeSeries
from safeds.exceptions import IllegalSchemaModificationError

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("table", "column_names", "expected"),
    [
        (
            TimeSeries._from_table(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
            ),
            ["feat1", "target", "time"],
            TimeSeries._from_table(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat1": [1, 2, 3],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
            ),
        ),
        (
            TimeSeries._from_table(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "other": [3, 4, 5],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
            ),
            ["feat1", "other", "target", "time"],
            TimeSeries._from_table(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat1": [1, 2, 3],
                        "other": [3, 4, 5],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
            ),
        ),
        (
            TimeSeries._from_table(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "other": [3, 4, 5],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
            ),
            ["feat1", "target", "time"],
            TimeSeries._from_table(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat1": [1, 2, 3],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
            ),
        ),
    ],
    ids=["keep_feature_and_target_column", "keep_non_feature_column", "don't_keep_non_feature_column"],
)
def test_should_return_table(table: TimeSeries, column_names: list[str], expected: TimeSeries) -> None:
    new_table = table.keep_only_columns(column_names)
    assert_that_time_series_are_equal(new_table, expected)


@pytest.mark.parametrize(
    ("table", "column_names", "error_msg"),
    [
        (
            TimeSeries._from_table(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "other": [3, 5, 7],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
                ["feat1", "feat2"],
            ),
            ["feat1", "feat2"],
            r"Illegal schema modification: Must keep the target column.",
        ),
        (
            TimeSeries._from_table(
                Table(
                    {
                        "time": [0, 1, 2],
                        "feat1": [1, 2, 3],
                        "feat2": [4, 5, 6],
                        "other": [3, 5, 7],
                        "target": [7, 8, 9],
                    },
                ),
                "target",
                "time",
                ["feat1", "feat2"],
            ),
            ["target", "feat1", "other"],
            r"Illegal schema modification: Must keep the time column.",
        ),
    ],
    ids=["table_remove_target", "table_remove_time"],
)
def test_should_raise_illegal_schema_modification(table: TimeSeries, column_names: list[str], error_msg: str) -> None:
    with pytest.raises(
        IllegalSchemaModificationError,
        match=error_msg,
    ):
        table.keep_only_columns(column_names)
