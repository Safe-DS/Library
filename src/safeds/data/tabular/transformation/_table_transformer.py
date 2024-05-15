from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table


class TableTransformer(ABC):
    """Learn a transformation for a set of columns in a `Table` and transform another `Table` with the same columns."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    # The decorator is needed so the class really cannot be instantiated
    @abstractmethod
    def __init__(self) -> None:
        self._column_names: list[str] | None = None

    # The decorator ensures that the method is overridden in all subclasses
    @abstractmethod
    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__qualname__,
            self._column_names,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._column_names is not None

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def fit(self, table: Table, column_names: list[str] | None) -> Self:
        """
        Learn a transformation for a set of columns in a table.

        **Note:** This transformer is not modified.

        Parameters
        ----------
        table:
            The table used to fit the transformer.
        column_names:
            The list of columns from the table used to fit the transformer. If `None`, all columns are used.

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
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """

    def fit_and_transform(
        self,
        table: Table,
        column_names: list[str] | None = None,
    ) -> tuple[Self, Table]:
        """
        Learn a transformation for a set of columns in a table and apply the learned transformation to the same table.

        **Note:** Neither this transformer nor the given table are modified.

        Parameters
        ----------
        table:
            The table used to fit the transformer. The transformer is then applied to this table.
        column_names:
            The list of columns from the table used to fit the transformer. If `None`, all columns are used.

        Returns
        -------
        fitted_transformer:
            The fitted transformer.
        transformed_table:
            The transformed table.
        """
        fitted_transformer = self.fit(table, column_names)
        transformed_table = fitted_transformer.transform(table)
        return fitted_transformer, transformed_table
