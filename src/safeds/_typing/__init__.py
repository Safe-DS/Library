from __future__ import annotations

import datetime
from decimal import Decimal
from typing import TypeAlias

from safeds.data.tabular.containers import Cell

# Literals
_NumericLiteral: TypeAlias = int | float | Decimal
_TemporalLiteral: TypeAlias = datetime.date | datetime.time | datetime.datetime | datetime.timedelta
_PythonLiteral: TypeAlias = _NumericLiteral | bool | str | bytes | _TemporalLiteral

# Convertible to cell (we cannot restrict `Cell`, because `Row.get_cell` returns a `Cell[Any]`)
_ConvertibleToCell: TypeAlias = _PythonLiteral | Cell | None
_ConvertibleToBooleanCell: TypeAlias = bool | Cell | None
_ConvertibleToIntCell: TypeAlias = int | Cell | None
_ConvertibleToStringCell: TypeAlias = str | Cell | None


__all__ = [
    "_ConvertibleToBooleanCell",
    "_ConvertibleToCell",
    "_ConvertibleToIntCell",
    "_ConvertibleToStringCell",
    "_NumericLiteral",
    "_PythonLiteral",
    "_TemporalLiteral",
]
