import polars as pl
import pytest

from safeds._utils import _safe_collect_lazy_frame, _safe_collect_lazy_frame_schema
from safeds.exceptions import LazyComputationError


def test_safe_collect_lazy_frame() -> None:
    frame = pl.LazyFrame().select("a")
    with pytest.raises(LazyComputationError):
        _safe_collect_lazy_frame(frame)


def test_safe_collect_lazy_frame_schema() -> None:
    frame = pl.LazyFrame().select("a")
    with pytest.raises(LazyComputationError):
        _safe_collect_lazy_frame_schema(frame)
