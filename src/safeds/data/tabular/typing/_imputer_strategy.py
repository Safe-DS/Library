from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from safeds._utils import _structural_hash

if TYPE_CHECKING:
    from sklearn.impute import SimpleImputer as sk_SimpleImputer


class ImputerStrategy(ABC):
    """
    The abstract base class of the different imputation strategies supported by the `Imputer`.

    This class is only needed for type annotations. Use the subclasses nested inside `Imputer.Strategy` instead.
    """

    @abstractmethod
    def _augment_imputer(self, imputer: sk_SimpleImputer) -> None:
        """
        Set the imputer strategy of the given imputer.

        Parameters
        ----------
        imputer:
            The imputer to augment.
        """

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Compare two imputer strategies.

        Parameters
        ----------
        other:
            other object to compare to

        Returns
        -------
        equals:
            Whether the two imputer strategies are equal
        """

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this imputer strategy.

        Returns
        -------
        hash:
            The hash value.
        """
        return _structural_hash(self.__class__.__qualname__)
