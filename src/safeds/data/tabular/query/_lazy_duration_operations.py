from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from safeds._utils import _structural_hash
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import DurationOperations

if TYPE_CHECKING:
    import polars as pl

    from safeds.data.tabular.containers import Cell


class _LazyDurationOperations(DurationOperations):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, expression: pl.Expr) -> None:
        self._expression: pl.Expr = expression

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LazyDurationOperations):
            return NotImplemented
        if self is other:
            return True
        return self._expression.meta.eq(other._expression)

    def __hash__(self) -> int:
        return _structural_hash(self._expression.meta.serialize())

    def __repr__(self) -> str:
        return f"_LazyDurationOperations({self._expression})"

    def __sizeof__(self) -> int:
        return self._expression.__sizeof__()

    def __str__(self) -> str:
        return f"({self._expression}).dur"

    # ------------------------------------------------------------------------------------------------------------------
    # Duration operations
    # ------------------------------------------------------------------------------------------------------------------

    def abs(self) -> Cell[None]:
        return _LazyCell(self._expression.abs())

    def full_weeks(self) -> Cell[int | None]:
        import polars as pl

        # We must round towards zero
        return _LazyCell((self._expression.dt.total_days() / 7).cast(pl.Int64()))

    def full_days(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.total_days())

    def full_hours(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.total_hours())

    def full_minutes(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.total_minutes())

    def full_seconds(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.total_seconds())

    def full_milliseconds(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.total_milliseconds())

    def full_microseconds(self) -> Cell[int | None]:
        return _LazyCell(self._expression.dt.total_microseconds())

    def to_string(
        self,
        *,
        format: Literal["iso", "pretty"] = "iso",
    ) -> Cell[str | None]:
        polars_format = "iso" if format == "iso" else "polars"

        return _LazyCell(self._expression.dt.to_string(polars_format))
