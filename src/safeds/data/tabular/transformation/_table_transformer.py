from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table
    from safeds.data.tabular.typing import Schema


class TableTransformer(ABC):
    """Learn a transformation for a set of columns in a `Table` and transform another `Table` with the same columns."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    # The decorator is needed so the class really cannot be instantiated
    @abstractmethod
    def __init__(self) -> None:
        # Schema of input table
        self._input: Schema | None = None

        # Schema of added columns
        self._added: Schema | None = None

        # Map of column names to the schema of their replacements
        self._replaced: dict[str, Schema] | None = None

        # Names of columns that were removed
        self._removed: list[str] | None = None

    # The decorator ensures that the method is overridden in all subclasses
    @abstractmethod
    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__qualname__,
            self._input,
            self._added,
            self._replaced,
            self._removed,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return None not in (self._input, self._added, self._replaced, self._removed)

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def fit(self, table: Table, column_names: list[str] | None) -> Self:
        """
        Learn a transformation for a set of columns in a table.

        This transformer is not modified.

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

        The table is not modified.

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

        Neither the transformer nor the table are modified.

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

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _check_additional_fit_preconditions(self, table: Table) -> None:  # noqa: B027
        """
        Check additional preconditions for fitting the transformer and raise an error if any are violated.

        Parameters
        ----------
        table:
            The table used to fit the transformer.
        """

    def _check_additional_transform_preconditions(self, table: Table) -> None:  # noqa: B027
        """
        Check additional preconditions for transforming with the transformer and raise an error if any are violated.

        Parameters
        ----------
        table:
            The table to which the learned transformation is applied.
        """

    @abstractmethod
    def _clone(self) -> Self:
        """
        Return a new instance of this transformer with the same settings.

        Returns
        -------
        clone:
            A new instance of this transformer.
        """
