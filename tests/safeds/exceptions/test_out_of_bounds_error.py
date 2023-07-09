import re

import pytest
from safeds.exceptions import Bound, ClosedBound, OpenBound, OutOfBoundsError, _Infinity, _MinInfinity


@pytest.mark.parametrize(
    "actual",
    [0, 1, -1, 2, -2, float("inf"), float("-inf")],
    ids=["0", "1", "-1", "2", "-2", "inf", "-inf"],
)
@pytest.mark.parametrize(
    ("lower_bound", "match_lower"),
    [
        (ClosedBound(-1), "[-1"),
        (OpenBound(-1), "(-1"),
        (None, "(-\u221e"),
    ],
    ids=["lb_closed_-1", "lb_open_-1", "lb_none"],
)
@pytest.mark.parametrize(
    ("upper_bound", "match_upper"),
    [
        (ClosedBound(1), "1]"),
        (OpenBound(1), "1)"),
        (None, "\u221e)"),
    ],
    ids=["ub_closed_-1", "ub_open_-1", "ub_none"],
)
def test_should_raise_out_of_bounds_error(
    actual: float,
    lower_bound: Bound | None,
    upper_bound: Bound | None,
    match_lower: str,
    match_upper: str,
) -> None:
    # Check (-inf, inf) interval:
    if lower_bound is None and upper_bound is None:
        with pytest.raises(
            ValueError,
            match=r"Illegal interval: Attempting to raise OutOfBoundsError, but no bounds given\.",
        ):
            raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)
        return
    # All tests: Check interval where lower > upper:
    if lower_bound is not None and upper_bound is not None:
        with pytest.raises(
            ValueError,
            match=r"Illegal interval: Attempting to raise OutOfBoundsError, but upper bound is less than the lower "
                  r"bound\.",
        ):
            raise OutOfBoundsError(actual, lower_bound=upper_bound, upper_bound=lower_bound)
    # Check case where actual value lies inside the interval:
    if (lower_bound is None or lower_bound.check_lower_bound(actual)) and (
        upper_bound is None or upper_bound.check_upper_bound(actual)
    ):
        with pytest.raises(
            ValueError,
            match=r"Illegal interval: Attempting to raise OutOfBoundsError, but value is not out of bounds\.",
        ):
            raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)
        return
    # Check that error is raised correctly:
    with pytest.raises(
        OutOfBoundsError,
        match=rf"{actual} is not inside {re.escape(match_lower)}, {re.escape(match_upper)}.",
    ):
        raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)


@pytest.mark.parametrize(
    ("value", "expected_value", "bound", "lower_bound"),
    [
        (2, True, ClosedBound(2), False),
        (2, True, ClosedBound(2), True),
        (2, True, ClosedBound(3), False),
        (2, True, ClosedBound(1), True),
        (2, False, OpenBound(2), False),
        (2, False, OpenBound(2), True),
        (2, False, OpenBound(1), False),
        (2, False, OpenBound(3), True),
        (2, False, _Infinity(), True),
        (2, True, _Infinity(), False),
        (2, True, _MinInfinity(), True),
        (2, False, _MinInfinity(), False),
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
    value: float,
    expected_value: bool,
    bound: Bound,
    lower_bound: bool,
) -> None:
    if lower_bound:
        assert expected_value == bound.check_lower_bound(value)
    else:
        assert expected_value == bound.check_upper_bound(value)
