import re

import pytest
from safeds.exceptions import Bound, ClosedBound, Infinity, MinInfinity, OpenBound, OutOfBoundsError


@pytest.mark.parametrize(
    "actual", [0, 1, -1, 2, -2, float("inf"), float("-inf")], ids=["0", "1", "-1", "2", "-2", "inf", "-inf"],
)
@pytest.mark.parametrize(
    ("lower_bound", "match_lower"),
    [
        (ClosedBound(-1), "[-1"),
        (OpenBound(-1), "(-1"),
        (MinInfinity(), "(-\u221e"),
        (None, "(-\u221e"),
    ],
    ids=["lb_closed_-1", "lb_open_-1", "lb_open_inf", "lb_none"],
)
@pytest.mark.parametrize(
    ("upper_bound", "match_upper"),
    [
        (ClosedBound(1), "1]"),
        (OpenBound(1), "1)"),
        (Infinity(), "\u221e)"),
        (None, "\u221e)"),
    ],
    ids=["ub_closed_-1", "ub_open_-1", "ub_open_inf", "ub_none"],
)
def test_should_raise_in_out_of_bounds_error(
    actual: float,
    lower_bound: Bound | None,
    upper_bound: Bound | None,
    match_lower: str,
    match_upper: str,
) -> None:
    if (lower_bound is None or isinstance(lower_bound, MinInfinity)) and (
        upper_bound is None or isinstance(upper_bound, Infinity)
    ):
        with pytest.raises(NotImplementedError, match=r"Value cannot be out of bounds if there are no bounds."):
            raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)
    elif lower_bound is not None and upper_bound is not None and upper_bound._value < lower_bound._value:
        with pytest.raises(NotImplementedError, match=r"The upper bound cannot be less than the lower bound."):
            raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)
    elif lower_bound is not None and not lower_bound._cmp_lower_bound(actual):
        with pytest.raises(NotImplementedError, match=r"The value should not be lower than the interval."):
            raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)
    elif upper_bound is not None and not upper_bound._cmp_upper_bound(actual):
        with pytest.raises(NotImplementedError, match=r"The value should not be larger than the interval."):
            raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)
    else:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"{actual} is not inside {re.escape(match_lower)}, {re.escape(match_upper)}.",
        ):
            raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)


@pytest.mark.parametrize(
    ("value", "expected_value", "bound", "lower_bound"),
    [
        (2, False, ClosedBound(2), False),
        (2, False, ClosedBound(2), True),
        (2, True, ClosedBound(3), False),
        (2, True, ClosedBound(1), True),
        (2, True, OpenBound(2), False),
        (2, True, OpenBound(2), True),
        (2, False, OpenBound(1), False),
        (2, False, OpenBound(3), True),
        (2, False, Infinity(), True),
        (2, True, Infinity(), False),
        (2, True, MinInfinity(), True),
        (2, False, MinInfinity(), False),
    ],
    ids=[
        "ex_false-close_2-upper",
        "ex_false-close_2-lower",
        "ex_true-close_3-upper",
        "ex_true-close_1-lower",
        "ex_true-open_2-upper",
        "ex_true-open_2-lower",
        "ex_false-open_1-upper",
        "ex_false-open_3-lower",
        "ex_false-inf-lower",
        "ex_true-inf-upper",
        "ex_true--inf-lower",
        "ex_false--inf-upper",
    ],
)
def test_should_return_true_if_value_in_bounds(
    value: float, expected_value: bool, bound: Bound, lower_bound: bool,
) -> None:
    if lower_bound:
        assert expected_value == bound._cmp_lower_bound(value)
    else:
        assert expected_value == bound._cmp_upper_bound(value)
