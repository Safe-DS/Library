from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table


class TableTransformer(ABC):
    """
    Learn a transformation for a set of columns in a `Table` and transform another `Table` with the same columns.

    Parameters
    ----------
    selector:
        The list of columns used to fit the transformer. If `None`, all suitable columns are used.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    # The decorator is needed so the class really cannot be instantiated
    @abstractmethod
    def __init__(self, selector: str | list[str] | None) -> None:
        if isinstance(selector, str):
            selector = [selector]

        self._selector: list[str] | None = selector

    # The decorator ensures that the method is overridden in all subclasses
    @abstractmethod
    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__qualname__,
            self._selector,
            self.is_fitted,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def fit(self, table: Table) -> Self:
        """
        Learn a transformation for a set of columns in a table.

        **Note:** This transformer is not modified.

        Parameters
        ----------
        table:
            The table used to fit the transformer.

        Returns
        -------
        fitted_transformer:
            The fitted transformer.
        """

    @abstractmethod
    def transform(self, table: Table) -> Table:
        """
        Apply the learned transformation to a table.

        **Note:** The given table is not modified.

        Parameters
        ----------
        table:
            The table to which the learned transformation is applied.

        Returns
        -------
        transformed_table:
            The transformed table.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted yet.
        """

    def fit_and_transform(self, table: Table) -> tuple[Self, Table]:
        """
        Learn a transformation for a set of columns in a table and apply the learned transformation to the same table.

        **Note:** Neither this transformer nor the given table are modified.

        Parameters
        ----------
        table:
            The table used to fit the transformer. The transformer is then applied to this table.

        Returns
        -------
        fitted_transformer:
            The fitted transformer.
        transformed_table:
            The transformed table.
        """
        fitted_transformer = self.fit(table)
        transformed_table = fitted_transformer.transform(table)
        return fitted_transformer, transformed_table
