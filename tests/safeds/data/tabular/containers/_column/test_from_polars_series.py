import polars as pl
import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pl.Series([]), []),
        (pl.Series([True, False, True]), [True, False, True]),
        (pl.Series([1, 2, 3]), [1, 2, 3]),
        (pl.Series([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]),
        (pl.Series(["a", "b", "c"]), ["a", "b", "c"]),
    ],
    ids=["empty", "boolean", "integer", "real number", "string"],
)
def test_should_store_the_data(series: pl.Series, expected: Column) -> None:
    assert list(Column._from_polars_series(series)) == expected

# TODO
# @pytest.mark.parametrize(
#     ("series", "type_"),
#     [
#         (pd.Series([True, False, True]), Boolean()),
#         (pd.Series([1, 2, 3]), Boolean()),
#     ],
#     ids=["type is correct", "type is wrong"],
# )
# def test_should_use_type_if_passed(series: pd.Series, type_: ColumnType) -> None:
#     assert Column._from_polars_series(series, type_).type == type_
#
#
# @pytest.mark.parametrize(
#     ("series", "expected"),
#     [
#         (pd.Series([]), Nothing()),
#         (pd.Series([True, False, True]), Boolean()),
#         (pd.Series([1, 2, 3]), Integer()),
#         (pd.Series([1.0, 2.0, 3.0]), Integer()),
#         (pd.Series([1.0, 2.5, 3.0]), RealNumber()),
#         (pd.Series(["a", "b", "c"]), String()),
#         (pd.Series([1, 2.0, "a", True]), Anything(is_nullable=False)),
#     ],
#     ids=["empty", "boolean", "integer", "real number .0", "real number", "string", "mixed"],
# )
# def test_should_infer_type_if_not_passed(series: pd.Series, expected: ColumnType) -> None:
#     assert Column._from_polars_series(series).type == expected
