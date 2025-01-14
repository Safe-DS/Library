from __future__ import annotations

import datetime
from decimal import Decimal
from typing import TypeAlias

from safeds.data.tabular.containers import Cell

_NumericLiteral: TypeAlias = int | float | Decimal
_TemporalLiteral: TypeAlias = datetime.date | datetime.time | datetime.datetime | datetime.timedelta
_PythonLiteral: TypeAlias = _NumericLiteral | bool | str | bytes | _TemporalLiteral
_ConvertibleToCell: TypeAlias = _PythonLiteral | Cell | None

__all__ = [
    "_ConvertibleToCell",
    "_NumericLiteral",
    "_PythonLiteral",
    "_TemporalLiteral",
]
