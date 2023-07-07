from __future__ import annotations

from abc import ABC, abstractmethod


class OutOfBoundsError(ValueError):
    """
    A generic exception to signal that a (float) value is out of bounds.

    Parameters
    ----------
    actual: float
        The actual value.
    lower_bound: _Bound | None
        The lower Bound.
    upper_bound: _Bound | None
        The lower Bound.
    """

    def __init__(self, actual: float, *, lower_bound: _Bound | None = None, upper_bound: _Bound | None = None):
        if lower_bound is None and upper_bound is None:
            raise NotImplementedError("Value cannot be out of bounds if there are no bounds.")
        super().__init__(f"{actual} is not inside {lower_bound._str_lower_bound()}, {upper_bound._str_upper_bound()}.")

    class _Bound(ABC):
        def __init__(self, value: float):
            self._value = value

        def __str__(self) -> str:
            return str(self._value)

        @abstractmethod
        def _str_lower_bound(self) -> str:
            pass

        @abstractmethod
        def _str_upper_bound(self) -> str:
            pass

    class _ClosedBound(_Bound):
        def __init__(self, value: float):
            super().__init__(value)

        def _str_lower_bound(self) -> str:
            return f"[{self}"

        def _str_upper_bound(self) -> str:
            return f"{self}]"

    class _OpenBound(_Bound):
        def __init__(self, value: float):
            super().__init__(value)

        def _str_lower_bracket(self) -> str:
            return f"({self}"

        def _str_upper_bracket(self) -> str:
            return f"{self})"

    class _Infinity(_OpenBound):

        def __init__(self):
            super().__init__(float("nan"))

        def __str__(self) -> str:
            return "\u221e"
