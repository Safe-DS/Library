from __future__ import annotations

import datetime
from decimal import Decimal
from typing import TypeAlias

from safeds.data.tabular.containers import Cell

_NumericLiteral: TypeAlias = int | float | Decimal
_TemporalLiteral: TypeAlias = datetime.date | datetime.time | datetime.datetime | datetime.timedelta
_PythonLiteral: TypeAlias = _NumericLiteral | bool | str | bytes | _TemporalLiteral
_ConvertibleToCell: TypeAlias = _PythonLiteral | Cell | None
_BooleanCell: TypeAlias = Cell[bool | None]
# We cannot restrict `Cell`, because `Row.get_cell` returns a `Cell[Any]`.
_ConvertibleToBooleanCell: TypeAlias = bool | Cell | None


__all__ = [
    "_BooleanCell",
    "_ConvertibleToBooleanCell",
    "_ConvertibleToCell",
    "_NumericLiteral",
    "_PythonLiteral",
    "_TemporalLiteral",
]
