import re

import pytest
from numpy import isnan
from safeds.exceptions import Bound, ClosedBound, OpenBound, OutOfBoundsError


@pytest.mark.parametrize(
    "actual",
    [0, 1, -1, 2, -2],
    ids=["0", "1", "-1", "2", "-2"],
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
            match=r"Illegal interval: Attempting to raise OutOfBoundsError, but given upper bound .+ is actually less "
            r"than given lower bound .+\.",
        ):
            raise OutOfBoundsError(actual, lower_bound=upper_bound, upper_bound=lower_bound)
    # Check case where actual value lies inside the interval:
    if (lower_bound is None or lower_bound._check_lower_bound(actual)) and (
        upper_bound is None or upper_bound._check_upper_bound(actual)
    ):
        with pytest.raises(
            ValueError,
            match=rf"Illegal interval: Attempting to raise OutOfBoundsError, but value {actual} is not actually outside"
            rf" given interval [\[(].+,.+[\])]\.",
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
        (2, False, OpenBound(float("inf")), True),
        (2, True, OpenBound(float("inf")), False),
        (2, True, OpenBound(float("-inf")), True),
        (2, False, OpenBound(float("-inf")), False),
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
        assert expected_value == bound._check_lower_bound(value)
    else:
        assert expected_value == bound._check_upper_bound(value)


@pytest.mark.parametrize(
    "value", [float("nan"), float("-inf"), float("inf")], ids=["nan", "neg_inf", "inf"]
)
def test_should_raise_value_error(value: float) -> None:
    if isnan(value):
        with pytest.raises(ValueError, match="Bound must be a real number, not nan."):
            ClosedBound(value)
        with pytest.raises(ValueError, match="Bound must be a real number, not nan."):
            OpenBound(value)
    else:
        with pytest.raises(ValueError, match=r"ClosedBound must be a real number, not \+\/\-inf\."):
            ClosedBound(value)
