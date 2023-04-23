import pandas as pd
import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import Boolean, String, Integer, RealNumber, ColumnType


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pd.Series([]), []),
        (pd.Series([True, False, True]), [True, False, True]),
        (pd.Series([1, 2, 3]), [1, 2, 3]),
        (pd.Series([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]),
        (pd.Series(["a", "b", "c"]), ["a", "b", "c"]),
        (pd.Series([1, 2.0, "a", True]), [1, 2.0, "a", True]),
    ],
    ids=["empty", "boolean", "integer", "real number", "string", "mixed"],
)
def test_should_store_the_data(series: pd.Series, expected: Column) -> None:
    assert list(Column._from_pandas_series(series)) == expected


@pytest.mark.parametrize(
    ("series", "type_"),
    [
        (pd.Series([True, False, True]), Boolean()),
        (pd.Series([1, 2, 3]), Boolean()),
    ],
    ids=["type is correct", "type is wrong"],
)
def test_should_use_type_if_passed(series: pd.Series, type_: ColumnType) -> None:
    assert Column._from_pandas_series(series, type_).type == type_


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pd.Series([]), String()),
        (pd.Series([True, False, True]), Boolean()),
        (pd.Series([1, 2, 3]), Integer()),
        (pd.Series([1.0, 2.0, 3.0]), RealNumber()),
        (pd.Series(["a", "b", "c"]), String()),
        (pd.Series([1, 2.0, "a", True]), String()),
    ],
    ids=["empty", "boolean", "integer", "real number", "string", "mixed"],
)
def test_should_infer_type_if_not_passed(series: pd.Series, expected: ColumnType) -> None:
    assert Column._from_pandas_series(series).type == expected
