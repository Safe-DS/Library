from __future__ import annotations

from abc import ABC, abstractmethod


class OutOfBoundsError(ValueError):
    """
    A generic exception that can be used to signal that a (float) value is outside its expected range.

    Parameters
    ----------
    actual: float
        The actual value.
    lower_bound: Bound | None
        The lower Bound.
    upper_bound: Bound | None
        The upper Bound.
    """

    def __init__(self, actual: float, *, lower_bound: Bound | None = None, upper_bound: Bound | None = None):
        if (lower_bound is None or isinstance(lower_bound, MinInfinity)) and (
            upper_bound is None or isinstance(upper_bound, Infinity)
        ):
            raise NotImplementedError("Value cannot be out of bounds if there are no bounds.")
        if lower_bound is None:
            lower_bound = MinInfinity()
        if upper_bound is None:
            upper_bound = Infinity()
        if upper_bound._value < lower_bound._value:
            raise NotImplementedError("The upper bound cannot be less than the lower bound.")
        if not lower_bound.check_lower_bound(actual):
            raise NotImplementedError("The value should not be lower than the interval.")
        if not upper_bound.check_upper_bound(actual):
            raise NotImplementedError("The value should not be larger than the interval.")
        super().__init__(f"{actual} is not inside {lower_bound.str_lower_bound()}, {upper_bound.str_upper_bound()}.")


class Bound(ABC):
    """Abstract base class for (lower or upper) Bounds on a float value."""

    def __init__(self, value: float):
        self._value = value

    def __str__(self) -> str:
        """Get a string representation of the concrete value of the Bound."""
        return str(self._value)

    @abstractmethod
    def str_lower_bound(self) -> str:
        """Get a string representation of the Bound as the lower Bound of an interval."""

    @abstractmethod
    def str_upper_bound(self) -> str:
        """Get a string representation of the Bound as the upper Bound of an interval."""

    @abstractmethod
    def check_lower_bound(self, value: float) -> bool:
        """Check that a value does not exceed the Bound on the lower side."""

    @abstractmethod
    def check_upper_bound(self, value: float) -> bool:
        """Check that a value does not exceed the Bound on the upper side."""


class ClosedBound(Bound):
    """A closed Bound, i.e. the value on the border belongs to the range."""

    def __init__(self, value: float):
        super().__init__(value)

    def str_lower_bound(self) -> str:
        """Get a string representation of the Bound as the lower Bound of an interval."""
        return f"[{self}"

    def str_upper_bound(self) -> str:
        """Get a string representation of the Bound as the upper Bound of an interval."""
        return f"{self}]"

    def check_lower_bound(self, value: float) -> bool:
        """Check that a value does not exceed the Bound on the lower side."""
        return value > self._value

    def check_upper_bound(self, value: float) -> bool:
        """Check that a value does not exceed the Bound on the upper side."""
        return value < self._value


class OpenBound(Bound):
    """
    An open Bound, i.e. the value on the border does not belong to the range.

    May be infinite (unbounded).
    """

    def __init__(self, value: float):
        super().__init__(value)

    def str_lower_bound(self) -> str:
        """Get a string representation of the Bound as the lower Bound of an interval."""
        return f"({self}"

    def str_upper_bound(self) -> str:
        """Get a string representation of the Bound as the upper Bound of an interval."""
        return f"{self})"

    def check_lower_bound(self, value: float) -> bool:
        """Check that a value does not exceed the Bound on the lower side."""
        return value >= self._value

    def check_upper_bound(self, value: float) -> bool:
        """Check that a value does not exceed the Bound on the upper side."""
        return value <= self._value


class Infinity(OpenBound):
    """An infinite or unrestricted upper Bound."""

    def __init__(self) -> None:
        super().__init__(float("inf"))

    def __str__(self) -> str:
        """Get a string representation of the concrete value of the Bound."""
        return "\u221e"


class MinInfinity(OpenBound):
    """An infinite or unrestricted lower Bound."""

    def __init__(self) -> None:
        super().__init__(float("-inf"))

    def __str__(self) -> str:
        """Get a string representation of the concrete value of the Bound."""
        return "-\u221e"
