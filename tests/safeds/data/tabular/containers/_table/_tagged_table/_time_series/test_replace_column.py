import pytest
from safeds.data.tabular.containers import Column, TimeSeries
from safeds.exceptions import IllegalSchemaModificationError

from tests.helpers import assert_that_time_series_are_equal


@pytest.mark.parametrize(
    ("original_table", "new_columns", "column_name_to_be_replaced", "result_table"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
                ["feature_old"],
            ),
            [Column("feature_new", [2, 1, 0])],
            "feature_old",
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_new": [2, 1, 0],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
                ["feature_new"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
                ["feature_old"],
            ),
            [Column("feature_new_a", [2, 1, 0]), Column("feature_new_b", [4, 2, 0])],
            "feature_old",
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_new_a": [2, 1, 0],
                    "feature_new_b": [4, 2, 0],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
                ["feature_new_a", "feature_new_b"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
                ["feature_old"],
            ),
            [Column("no_feature_new", [2, 1, 0])],
            "no_feature_old",
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "no_feature_new": [2, 1, 0],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
                ["feature_old"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
                ["feature_old"],
            ),
            [Column("no_feature_new_a", [2, 1, 0]), Column("no_feature_new_b", [4, 2, 0])],
            "no_feature_old",
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "no_feature_new_a": [2, 1, 0],
                    "no_feature_new_b": [4, 2, 0],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
                ["feature_old"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
                ["feature_old"],
            ),
            [Column("target_new", [2, 1, 0])],
            "target_old",
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_new": [2, 1, 0],
                },
                "target_new",
                "time",
                ["feature_old"],
            ),
        ),
        (
            TimeSeries(
                {
                    "time_old": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time_old",
                ["feature_old"],
            ),
            [Column("time_new", [1, 2, 3])],
            "time_old",
            TimeSeries(
                {
                    "time_new": [1, 2, 3],
                    "feature_old": [0, 1, 2],
                    "no_feature_old": [2, 3, 4],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time_new",
                ["feature_old"],
            ),
        ),
    ],
    ids=[
        "replace_feature_column_with_one",
        "replace_feature_column_with_multiple",
        "replace_non_feature_column_with_one",
        "replace_non_feature_column_with_multiple",
        "replace_target_column",
        "replace_time_column",
    ],
)
def test_should_replace_column(
    original_table: TimeSeries,
    new_columns: list[Column],
    column_name_to_be_replaced: str,
    result_table: TimeSeries,
) -> None:
    new_table = original_table.replace_column(column_name_to_be_replaced, new_columns)
    assert_that_time_series_are_equal(new_table, result_table)


@pytest.mark.parametrize(
    ("original_table", "new_columns", "column_name_to_be_replaced", "error"),
    [
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
            ),
            [],
            "target_old",
            'Target column "target_old" can only be replaced by exactly one new column.',
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
            ),
            [Column("target_new_a", [2, 1, 0]), Column("target_new_b"), [4, 2, 0]],
            "target_old",
            'Target column "target_old" can only be replaced by exactly one new column.',
        ),
        (
            TimeSeries(
                {
                    "time": [0, 1, 2],
                    "feature_old": [0, 1, 2],
                    "target_old": [3, 4, 5],
                },
                "target_old",
                "time",
            ),
            [Column("target_new_a", [2, 1, 0]), Column("target_new_b"), [4, 2, 0]],
            "time",
            'Time column "time" can only be replaced by exactly one new column.',
        ),
    ],
    ids=["zero_columns", "multiple_columns", "time_column"],
)
# here should be tested with time column as well but the test is weird to be extended
def test_should_throw_illegal_schema_modification(
    original_table: TimeSeries,
    new_columns: list[Column],
    column_name_to_be_replaced: str,
    error: str,
) -> None:
    with pytest.raises(
        IllegalSchemaModificationError,
        match=error,
    ):
        original_table.replace_column(column_name_to_be_replaced, new_columns)
