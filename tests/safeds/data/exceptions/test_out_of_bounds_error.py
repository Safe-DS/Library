import pytest
from safeds.exceptions import OutOfBoundsError, Bound, ClosedBound


@pytest.mark.parametrize(
    ("actual", "lower_bound", "upper_bound"),
    [
        (42, ClosedBound(-1), ClosedBound(1)),
    ],
    ids=["closed_closed"]
)
def test_should_raise_out_of_bounds_error(actual: float, lower_bound: Bound, upper_bound: Bound) -> None:
    #with pytest.raises(OutOfBoundsError):
    raise OutOfBoundsError(actual, lower_bound=lower_bound, upper_bound=upper_bound)


def test_should_raise_not_implemented_error() -> None:
    pass
