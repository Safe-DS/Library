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

    def __init__(self, actual: float, *, lower_bound: Bound = None, upper_bound: Bound = None):
        if (lower_bound is None or isinstance(lower_bound, MinInfinity)) and (upper_bound is None or isinstance(upper_bound, Infinity)):
            raise NotImplementedError("Value cannot be out of bounds if there are no bounds.")
        if lower_bound is None:
            lower_bound = MinInfinity()
        if upper_bound is None:
            upper_bound = Infinity()
        super().__init__(f"{actual} is not inside {lower_bound._str_lower_bound()}, {upper_bound._str_upper_bound()}.")


class Bound(ABC):
    def __init__(self, value: float):
        self._value = value

    def __str__(self) -> str:
        return str(self._value)

    def _is_float(self) -> bool:
        return True

    @abstractmethod
    def _str_lower_bound(self) -> str:
        pass

    @abstractmethod
    def _str_upper_bound(self) -> str:
        pass


class ClosedBound(Bound):
    def __init__(self, value: float):
        super().__init__(value)

    def _str_lower_bound(self) -> str:
        return f"[{self}"

    def _str_upper_bound(self) -> str:
        return f"{self}]"


class OpenBound(Bound):
    def __init__(self, value: float):
        super().__init__(value)

    def _str_lower_bound(self) -> str:
        return f"({self}"

    def _str_upper_bound(self) -> str:
        return f"{self})"


class Infinity(OpenBound):

    def __init__(self):
        super().__init__(float("nan"))

    def __str__(self) -> str:
        return "\u221e"

    def _is_float(self) -> bool:
        return False


class MinInfinity(OpenBound):

    def __init__(self):
        super().__init__(float("nan"))

    def __str__(self) -> str:
        return "-\u221e"

    def _is_float(self) -> bool:
        return False
