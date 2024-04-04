from abc import ABC, abstractmethod

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
        imputer: SimpleImputer
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

    @abstractmethod
    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this imputer strategy.

        Returns
        -------
        hash : int
            The hash value.
        """
