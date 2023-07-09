from __future__ import annotations

from abc import ABC, abstractmethod

from numpy import isnan


class OutOfBoundsError(ValueError):
    """A generic exception that can be used to signal that a (float) value is outside its expected range."""

    def __init__(self, actual: float, *, lower_bound: Bound | None = None, upper_bound: Bound | None = None):
        """
        Initialize an OutOfBoundsError.

        Parameters
        ----------
        actual: float
            The actual value that is outside its expected range.
        lower_bound: Bound | None
            The lower bound of the expected range.
        upper_bound: Bound | None
            The upper bound of the expected range.

        Raises
        ------
        ValueError
            If upper_bound < lower_bound or if actual does not lie outside the given interval.
        """
        if lower_bound is None and upper_bound is None:
            raise ValueError("Illegal interval: Attempting to raise OutOfBoundsError, but no bounds given.")
        # Use local variables with stricter types to help static analysis:
        _lower_bound: Bound = lower_bound if lower_bound is not None else OpenBound(float("-inf"))
        _upper_bound: Bound = upper_bound if upper_bound is not None else OpenBound(float("inf"))
        if _upper_bound.value < _lower_bound.value:
            raise ValueError(
                (
                    f"Illegal interval: Attempting to raise OutOfBoundsError, but given upper bound {_upper_bound} is "
                    f"actually less than given lower bound {_lower_bound}."
                ),
            )
        elif _lower_bound._check_lower_bound(actual) and _upper_bound._check_upper_bound(actual):
            raise ValueError(
                (
                    f"Illegal interval: Attempting to raise OutOfBoundsError, but value {actual} is not actually"
                    f" outside given interval {_lower_bound._str_lower_bound()}, {_upper_bound._str_upper_bound()}."
                ),
            )
        super().__init__(
            f"{actual} is not inside {_lower_bound._str_lower_bound()}, {_upper_bound._str_upper_bound()}.",
        )


class Bound(ABC):
    """Abstract base class for (lower or upper) Bounds on a float value."""

    def __init__(self, value: float):
        if isnan(value):
            raise ValueError("Bound must be a number or +/-infinity, not nan.")
        self._value = value

    def __str__(self) -> str:
        """Get a string representation of the concrete value of the Bound."""
        return str(self.value)

    @abstractmethod
    def _str_lower_bound(self) -> str:
        """Get a string representation of the Bound as the lower Bound of an interval."""

    @abstractmethod
    def _str_upper_bound(self) -> str:
        """Get a string representation of the Bound as the upper Bound of an interval."""

    @abstractmethod
    def _check_lower_bound(self, value: float) -> bool:
        """Check that a value does not exceed the Bound on the lower side."""

    @abstractmethod
    def _check_upper_bound(self, value: float) -> bool:
        """Check that a value does not exceed the Bound on the upper side."""

    @property
    def value(self) -> float:
        """Get the concrete value of the Bound."""
        return self._value


class ClosedBound(Bound):
    """A closed Bound, i.e. the value on the border belongs to the range."""

    def __init__(self, value: float):
        super().__init__(value)

    def _str_lower_bound(self) -> str:
        """Get a string representation of the ClosedBound as the lower Bound of an interval."""
        return f"[{self}"

    def _str_upper_bound(self) -> str:
        """Get a string representation of the ClosedBound as the upper Bound of an interval."""
        return f"{self}]"

    def _check_lower_bound(self, value: float) -> bool:
        """Check that a value is not strictly lower than the ClosedBound."""
        return value >= self.value

    def _check_upper_bound(self, value: float) -> bool:
        """Check that a value is not strictly higher than the ClosedBound."""
        return value <= self.value


class OpenBound(Bound):
    """
    An open Bound, i.e. the value on the border does not belong to the range.

    May be infinite (unbounded).
    """

    def __init__(self, value: float):
        super().__init__(value)

    def __str__(self) -> str:
        """Get a string representation of the concrete value of the Bound."""
        if self.value == float("-inf"):
            return "-\u221e"
        elif self.value == float("inf"):
            return "\u221e"
        else:
            return super().__str__()

    def _str_lower_bound(self) -> str:
        """Get a string representation of the OpenBound as the lower Bound of an interval."""
        return f"({self}"

    def _str_upper_bound(self) -> str:
        """Get a string representation of the OpenBound as the upper Bound of an interval."""
        return f"{self})"

    def _check_lower_bound(self, value: float) -> bool:
        """Check that a value is not lower or equal to the Bound."""
        return value > self.value

    def _check_upper_bound(self, value: float) -> bool:
        """Check that a value is not higher ot equal to the Bound."""
        return value < self.value
