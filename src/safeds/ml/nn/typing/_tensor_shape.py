from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Self

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound


class TensorShape(ABC):
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

    @classmethod
    def get_size(cls: type[Self], dimension: int | None = None) -> int:
        """
        Return the size of the tensor in the specified dimension.

        Parameters.
        ----------
        dimension:
            The dimension index for which the size is to be retrieved.

        Return
        -------
            int: The size of the tensor in the specified dimension.
        """
        _check_bounds("dimension",dimension, lower_bound=_ClosedBound(0))
        if(dimension >= cls.dimensionality):
            #TODO maybe add error message indicating that the dimension is out of range
            return 0        
        if(dimension is None):
            return cls._dims
        return cls._dims[dimension]
    
    @abstractmethod
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