from __future__ import annotations

from abc import ABC, abstractmethod

from numpy import isinf, isnan


class OutOfBoundsError(ValueError):
    """
    A generic exception that can be used to signal that a (float) value is outside its expected range.

    Parameters
    ----------
    actual: float
        The actual value that is outside its expected range.
    name: str | None
        The name of the offending variable.
    lower_bound: Bound | None
        The lower bound of the expected range. Use None if there is no lower Bound.
    upper_bound: Bound | None
        The upper bound of the expected range. Use None if there is no upper Bound.
    """

    def __init__(
        self,
        actual: float,
        *,
        name: str | None = None,
        lower_bound: Bound | None = None,
        upper_bound: Bound | None = None,
    ):
        """
        Initialize an OutOfBoundsError.

        Parameters
        ----------
        actual: float
            The actual value that is outside its expected range.
        name: str | None
            The name of the offending variable.
        lower_bound: Bound | None
            The lower bound of the expected range. Use None if there is no lower Bound.
        upper_bound: Bound | None
            The upper bound of the expected range. Use None if there is no upper Bound.

        Raises
        ------
        ValueError
            * If one of the given Bounds is +/-inf. (For infinite Bounds, pass None instead.)
            * If one of the given Bounds is nan.
            * If upper_bound < lower_bound.
            * If actual does not lie outside the given interval.
            * If actual is not a real number.
        """
        # Validate bound parameters:
        if lower_bound is None and upper_bound is None:
            raise ValueError("Illegal interval: Attempting to raise OutOfBoundsError, but no bounds given.")
        if (lower_bound is not None and isinf(lower_bound.value)) or (
            upper_bound is not None and isinf(upper_bound.value)
        ):
            raise ValueError("Illegal interval: Lower and upper bounds must be real numbers, or None if unbounded.")
        # Validate actual parameter:
        if isinf(actual) or isnan(actual):
            raise ValueError("Attempting to raise OutOfBoundsError with actual value not being a real number.")
        # Use local variables with stricter types to help static analysis:
        _lower_bound: Bound = lower_bound if lower_bound is not None else OpenBound(float("-inf"))
        _upper_bound: Bound = upper_bound if upper_bound is not None else OpenBound(float("inf"))
        # Check bounds:
        if _upper_bound.value < _lower_bound.value:
            raise ValueError(
                (
                    f"Illegal interval: Attempting to raise OutOfBoundsError, but given upper bound {_upper_bound} is "
                    f"actually less than given lower bound {_lower_bound}."
                ),
            )
        # Check that actual is indeed outside the interval:
        elif _lower_bound._check_lower_bound(actual) and _upper_bound._check_upper_bound(actual):
            raise ValueError(
                (
                    f"Illegal interval: Attempting to raise OutOfBoundsError, but value {actual} is not actually"
                    f" outside given interval {_lower_bound._str_lower_bound()}, {_upper_bound._str_upper_bound()}."
                ),
            )
        # Raise the actual exception:
        full_variable_name = actual if name is None else f"{name} (={actual})"
        super().__init__(
            f"{full_variable_name} is not inside {_lower_bound._str_lower_bound()}, {_upper_bound._str_upper_bound()}.",
        )


class Bound(ABC):
    """
    Abstract base class for (lower or upper) Bounds on a float value.

    Parameters
    ----------
    value: float
        The value of the Bound.
    """

    def __init__(self, value: float):
        """
        Initialize a Bound.

        Parameters
        ----------
        value: float
            The value of the Bound.

        Raises
        ------
        ValueError
            If value is nan or if value is +/-inf and the Bound type does not allow for infinite Bounds.
        """
        if isnan(value):
            raise ValueError("Bound must be a real number, not nan.")
        self._value = value

    def __str__(self) -> str:
        """Get a string representation of the concrete value of the Bound."""
        return str(self.value)

    @property
    def value(self) -> float:
        """Get the concrete value of the Bound."""
        return self._value

    @abstractmethod
    def _str_lower_bound(self) -> str:
        """Get a string representation of the Bound as the lower Bound of an interval."""

    @abstractmethod
    def _str_upper_bound(self) -> str:
        """Get a string representation of the Bound as the upper Bound of an interval."""

    @abstractmethod
    def _check_lower_bound(self, actual: float) -> bool:
        """
        Check that a value does not exceed the Bound on the lower side.

        Parameters
        ----------
        actual: float
            The actual value that should be checked for not exceeding the Bound.
        """

    @abstractmethod
    def _check_upper_bound(self, actual: float) -> bool:
        """
        Check that a value does not exceed the Bound on the upper side.

        Parameters
        ----------
        actual: float
            The actual value that should be checked for not exceeding the Bound.
        """


class ClosedBound(Bound):
    """
    A closed Bound, i.e. the value on the border belongs to the range.

    Parameters
    ----------
    value: float
        The value of the Bound.
    """

    def __init__(self, value: float):
        """
        Initialize a ClosedBound.

        Parameters
        ----------
        value: float
            The value of the ClosedBound.

        Raises
        ------
        ValueError
            If value is nan or if value is +/-inf.
        """
        if value == float("-inf") or value == float("inf"):
            raise ValueError("ClosedBound must be a real number, not +/-inf.")
        super().__init__(value)

    def _str_lower_bound(self) -> str:
        """Get a string representation of the ClosedBound as the lower Bound of an interval."""
        return f"[{self}"

    def _str_upper_bound(self) -> str:
        """Get a string representation of the ClosedBound as the upper Bound of an interval."""
        return f"{self}]"

    def _check_lower_bound(self, actual: float) -> bool:
        """
        Check that a value is not strictly lower than the ClosedBound.

        Parameters
        ----------
        actual: float
            The actual value that should be checked for not exceeding the Bound.
        """
        return actual >= self.value

    def _check_upper_bound(self, actual: float) -> bool:
        """
        Check that a value is not strictly higher than the ClosedBound.

        Parameters
        ----------
        actual: float
            The actual value that should be checked for not exceeding the Bound.
        """
        return actual <= self.value


class OpenBound(Bound):
    """
    An open Bound, i.e. the value on the border does not belong to the range.

    Parameters
    ----------
    value: float
        The value of the OpenBound.
    """

    def __init__(self, value: float):
        """
        Initialize an OpenBound.

        Parameters
        ----------
        value: float
            The value of the OpenBound.

        Raises
        ------
        ValueError
            If value is nan.
        """
        super().__init__(value)

    def __str__(self) -> str:
        """Get a string representation of the concrete value of the OpenBound."""
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

    def _check_lower_bound(self, actual: float) -> bool:
        """
        Check that a value is not lower or equal to the OpenBound.

        Parameters
        ----------
        actual: float
            The actual value that should be checked for not exceeding the Bound.
        """
        return actual > self.value

    def _check_upper_bound(self, actual: float) -> bool:
        """
        Check that a value is not higher or equal to the OpenBound.

        Parameters
        ----------
        actual: float
            The actual value that should be checked for not exceeding the Bound.
        """
        return actual < self.value
