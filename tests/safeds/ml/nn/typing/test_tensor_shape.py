import pytest
from safeds.exceptions import OutOfBoundsError
from safeds.ml.nn.typing import TensorShape


def test_dims() -> None:
    dims = [2, 3, 4]
    ts = TensorShape(dims)
    assert ts._dims == dims


def test_dimensionality() -> None:
    dims = [2, 3, 4]
    ts = TensorShape(dims)
    assert ts.dimensionality == 3


def test_get_size_no_dimension() -> None:
    dims = [2, 3, 4]
    ts = TensorShape(dims)
    assert ts.get_size() == dims


def test_get_size_specific_dimension() -> None:
    dims = [2, 3, 4]
    ts = TensorShape(dims)
    assert ts.get_size(0) == 2
    assert ts.get_size(1) == 3
    assert ts.get_size(2) == 4


def test_get_size_out_of_bounds_dimension() -> None:
    dims = [2, 3, 4]
    ts = TensorShape(dims)
    with pytest.raises(OutOfBoundsError):
        ts.get_size(-1)


def test_hash() -> None:
    dims = [2, 3, 4]
    ts1 = TensorShape(dims)
    ts2 = TensorShape(dims)
    assert hash(ts1) == hash(ts2)
