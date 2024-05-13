from __future__ import annotations

from abc import ABC, abstractmethod


def _check_bounds(
    name: str,
    actual: float | None,
    *,
    lower_bound: _Bound | None = None,
    upper_bound: _Bound | None = None,
) -> None:
    """
    Check that a value is within the expected range and raise an error if it is not.

    Parameters
    ----------
    actual:
        The actual value that should be checked.
    name:
        The name of the offending variable.
    lower_bound:
        The lower bound of the expected range. Use None if there is no lower Bound.
    upper_bound:
        The upper bound of the expected range. Use None if there is no upper Bound.

    Raises
    ------
    OutOfBoundsError:
        If the actual value is outside its expected range.
    """
    from safeds.exceptions import OutOfBoundsError  # circular import

    if actual is None:
        return  # Skip the check if the actual value is None (i.e., not provided).

    if lower_bound is None:
        lower_bound = _OpenBound(float("-inf"))
    if upper_bound is None:
        upper_bound = _OpenBound(float("inf"))

    if not lower_bound._check_as_lower_bound(actual) or not upper_bound._check_as_upper_bound(actual):
        message = _build_error_message(name, actual, lower_bound, upper_bound)
        raise OutOfBoundsError(message)


def _build_error_message(name: str, actual: float, lower_bound: _Bound, upper_bound: _Bound) -> str:
    range_ = f"{lower_bound._to_string_as_lower_bound()}, {upper_bound._to_string_as_upper_bound()}"
    return f"{name} must be in {range_} but was {actual}."


class _Bound(ABC):
    """Lower or upper bound of the legal range of a value."""

    @abstractmethod
    def _check_as_lower_bound(self, actual: float) -> bool:
        """
        Treat this bound as the lower bound and check that a value does not exceed it.

        Parameters
        ----------
        actual:
            The actual value to check.

        Returns
        -------
        in_bounds:
            Whether the actual value is within the expected range.
        """

    @abstractmethod
    def _check_as_upper_bound(self, actual: float) -> bool:
        """
        Treat this bound as the upper bound and check that a value does not exceed it.

        Parameters
        ----------
        actual:
            The actual value to check.

        Returns
        -------
        in_bounds:
            Whether the actual value is within the expected range.
        """

    @abstractmethod
    def _to_string_as_lower_bound(self) -> str:
        """Treat this bound as the lower bound and get a string representation."""

    @abstractmethod
    def _to_string_as_upper_bound(self) -> str:
        """Treat this bound as the upper bound and get a string representation."""


class _ClosedBound(_Bound):
    """
    A closed bound where the border value belongs to the range.

    Parameters
    ----------
    value:
        The border value of the bound.
    """

    def __init__(self, value: float):
        self.value: float = value

    def _check_as_lower_bound(self, actual: float) -> bool:
        return actual >= self.value

    def _check_as_upper_bound(self, actual: float) -> bool:
        return actual <= self.value

    def _to_string_as_lower_bound(self) -> str:
        return f"[{_float_to_string(self.value)}"

    def _to_string_as_upper_bound(self) -> str:
        return f"{_float_to_string(self.value)}]"


class _OpenBound(_Bound):
    """
    An open bound where the border value does not belong to the range.

    Parameters
    ----------
    value:
        The border value of the bound.
    """

    def __init__(self, value: float):
        self.value: float = value

    def _check_as_lower_bound(self, actual: float) -> bool:
        return actual > self.value

    def _check_as_upper_bound(self, actual: float) -> bool:
        return actual < self.value

    def _to_string_as_lower_bound(self) -> str:
        return f"({_float_to_string(self.value)}"

    def _to_string_as_upper_bound(self) -> str:
        return f"{_float_to_string(self.value)})"


def _float_to_string(value: float) -> str:
    if value == float("-inf"):
        return "-\u221e"
    elif value == float("inf"):
        return "\u221e"
    else:
        return str(value)
