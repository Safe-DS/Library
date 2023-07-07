import re

import pytest
from safeds.exceptions import Bound, ClosedBound, Infinity, MinInfinity, OpenBound, OutOfBoundsError


@pytest.mark.parametrize(
    ("actual", "lower_bound", "match_lower"),
    [
        (42, ClosedBound(-1), "[-1"),
        (42, OpenBound(-1), "(-1"),
        (42, MinInfinity(), "(-\u221e"),
        (42, None, "(-\u221e"),
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
    elif lower_bound is not None and upper_bound is not None and lower_bound._value > upper_bound._value:
        with pytest.raises(NotImplementedError, match=r"The upper bound cannot be less than the lower bound."):
            raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)
    else:
        with pytest.raises(
            OutOfBoundsError,
            match=rf"{actual} is not inside {re.escape(match_lower)}, {re.escape(match_upper)}.",
        ):
            raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)
