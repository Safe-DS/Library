from __future__ import annotations

from abc import ABC, abstractmethod


class OutOfBoundsError(ValueError):
    """A generic exception that can be used to signal that a (float) value is outside its expected range."""

    def __init__(self, actual: float, *, lower_bound: Bound | None = None, upper_bound: Bound | None = None):
        """
        Initialize an OutOfBoundError.

        Parameters
        ----------
        actual: float
            The actual value that is outside its expected range.
        lower_bound: Bound | None
            The lower bound of the expected range.
        upper_bound: Bound | None
            The upper bound of the expected range.
        """
        if lower_bound is None and upper_bound is None:
            raise ValueError("Illegal interval: Attempting to raise OutOfBoundsError, but no bounds given.")
        lower_bound: Bound = lower_bound if lower_bound is not None else _MinInfinity()
        upper_bound: Bound = upper_bound if upper_bound is not None else _Infinity()
        if upper_bound.value < lower_bound.value:
            raise ValueError(
                "Illegal interval: Attempting to raise OutOfBoundsError, but upper bound is less than the lower bound.",
            )
        elif lower_bound.check_lower_bound(actual) and upper_bound.check_upper_bound(actual):
            raise ValueError(
                "Illegal interval: Attempting to raise OutOfBoundsError, but value is not out of bounds.",
            )
        super().__init__(f"{actual} is not inside {lower_bound.str_lower_bound()}, {upper_bound.str_upper_bound()}.")


class Bound(ABC):
    """Abstract base class for (lower or upper) Bounds on a float value."""

    def __init__(self, value: float):
        self._value = value

    def __str__(self) -> str:
        """Get a string representation of the concrete value of the Bound."""
        return str(self.value)

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

    @property
    def value(self) -> float:
        """Get the concrete value of the Bound."""
        return self._value


class ClosedBound(Bound):
    """A closed Bound, i.e. the value on the border belongs to the range."""

    def __init__(self, value: float):
        super().__init__(value)

    def str_lower_bound(self) -> str:
        """Get a string representation of the ClosedBound as the lower Bound of an interval."""
        return f"[{self}"

    def str_upper_bound(self) -> str:
        """Get a string representation of the ClosedBound as the upper Bound of an interval."""
        return f"{self}]"

    def check_lower_bound(self, value: float) -> bool:
        """Check that a value is not strictly lower than the ClosedBound."""
        return value >= self._value

    def check_upper_bound(self, value: float) -> bool:
        """Check that a value is not strictly higher than the ClosedBound."""
        return value <= self._value


class OpenBound(Bound):
    """
    An open Bound, i.e. the value on the border does not belong to the range.

    May be infinite (unbounded).
    """

    def __init__(self, value: float):
        super().__init__(value)

    def str_lower_bound(self) -> str:
        """Get a string representation of the OpenBound as the lower Bound of an interval."""
        return f"({self}"

    def str_upper_bound(self) -> str:
        """Get a string representation of the OpenBound as the upper Bound of an interval."""
        return f"{self})"

    def check_lower_bound(self, value: float) -> bool:
        """Check that a value is not lower or equal to the Bound."""
        return value > self._value

    def check_upper_bound(self, value: float) -> bool:
        """Check that a value is not higher ot equal to the Bound."""
        return value < self._value


class _Infinity(OpenBound):
    """An infinite or unrestricted upper Bound."""

    def __init__(self) -> None:
        super().__init__(float("inf"))

    def __str__(self) -> str:
        """Get a string representation of the concrete value of the Bound."""
        return "\u221e"

    def check_lower_bound(self, _: float) -> bool:
        """Check that a value does not exceed the unrestricted Bound on the lower side. Always False."""
        return False

    def check_upper_bound(self, _: float) -> bool:
        """Check that a value does not exceed the unrestricted Bound on the upper side. Always true."""
        return True


class _MinInfinity(OpenBound):
    """An infinite or unrestricted lower Bound."""

    def __init__(self) -> None:
        super().__init__(float("-inf"))

    def __str__(self) -> str:
        """Get a string representation of the concrete value of the Bound."""
        return "-\u221e"

    def check_lower_bound(self, _: float) -> bool:
        """Check that a value does not exceed the unrestricted Bound on the lower side. Always true."""
        return True

    def check_upper_bound(self, _: float) -> bool:
        """Check that a value does not exceed the unrestricted Bound on the upper side. Always false."""
        return False
