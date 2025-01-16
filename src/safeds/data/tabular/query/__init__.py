"""Classes that represent queries on the data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._datetime_operations import DatetimeOperations
    from ._duration_operations import DurationOperations
    from ._math_operations import MathOperations
    from ._string_operations import StringOperations

apipkg.initpkg(
    __name__,
    {
        "DatetimeOperations": "._datetime_operations:DatetimeOperations",
        "DurationOperations": "._duration_operations:DurationOperations",
        "MathOperations": "._math_operations:MathOperations",
        "StringOperations": "._string_operations:StringOperations",
    },
)

__all__ = [
    "DatetimeOperations",
    "DurationOperations",
    "MathOperations",
    "StringOperations",
]
