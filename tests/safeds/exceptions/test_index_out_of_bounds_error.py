import pytest

from safeds.exceptions import IndexOutOfBoundsError


@pytest.mark.parametrize(
    ["index", "match"],
    [
        (3, r"There is no element at index '3'."),
        ([3], r"There is no element at index '3'."),
        ([3, 5], r"There are no elements at indices \[3, 5\]."),
        (slice(3, 5), r"There is no element in the range \[3, 5\]."),
    ],
    ids=["int", "list", "list-size-1", "slice"]
)
def test_should_raise_index_out_of_bounds_error(index: int | list[int] | slice, match: str) -> None:
    with pytest.raises(IndexOutOfBoundsError, match=match):
        raise IndexOutOfBoundsError(index)
