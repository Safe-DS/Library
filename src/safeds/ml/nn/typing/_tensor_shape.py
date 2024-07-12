from __future__ import annotations

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound


class TensorShape:
    """
    Initializes a TensorShape object with the given dimensions.

    Parameters
    ----------
    dims: 
        A list of integers where each integer represents 
        the size of the tensor in a particular dimension.
    """
    
    def __init__(self, dims: list[int]) -> None:
        self._dims = dims 

    def get_size(self, dimension: int | None = None) -> int:
        """
        Return the size of the tensor in the specified dimension.

        Parameters.
        ----------
        dimension:
            The dimension index for which the size is to be retrieved.

        Returns
        -------
            int: The size of the tensor in the specified dimension.

        Raises
        ------
        OutOfBoundsError:
            If the actual value is outside its expected range.
        """
        _check_bounds("dimension",dimension, lower_bound=_ClosedBound(0))
        if dimension is not None and dimension >= self.dimensionality:
            #TODO maybe add error message indicating that the dimension is out of range
            return 0        
        if(dimension is None):
            return self._dims[0]
        return self._dims[dimension]
    
    def __hash__(self) -> int:
        return _structural_hash(self._dims)
     
    @property
    def dimensionality(self) -> int:
        """
        Returns the number of dimensions of the tensor.

        Returns
        -------
            int: The number of dimensions of the tensor.
        """
        return len(self._dims)