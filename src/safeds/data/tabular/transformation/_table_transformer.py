from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash

if TYPE_CHECKING:
    from safeds.data.tabular.containers import ExperimentalTable


class ExperimentalTableTransformer(ABC):
    """Learn a transformation for a set of columns in a `Table` and transform another `Table` with the same columns."""

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for a table transformer.

        Returns
        -------
        hash:
            The hash value.
        """
        added = self.get_names_of_added_columns() if self.is_fitted else []
        changed = self.get_names_of_changed_columns() if self.is_fitted else []
        removed = self.get_names_of_removed_columns() if self.is_fitted else []
        return _structural_hash(self.__class__.__qualname__, self.is_fitted, added, changed, removed)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""

    # ------------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def fit(self, table: ExperimentalTable, column_names: list[str] | None) -> Self:
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
    def transform(self, table: ExperimentalTable) -> ExperimentalTable:
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

    @abstractmethod
    def get_names_of_added_columns(self) -> list[str]:
        """
        Get the names of all new columns that have been added by the transformer.

        Returns
        -------
        added_columns:
            A list of names of the added columns, ordered as they will appear in the table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """

    @abstractmethod
    def get_names_of_changed_columns(self) -> list[str]:
        """
         Get the names of all columns that have been changed by the transformer.

        Returns
        -------
        changed_columns:
             A list of names of changed columns, ordered as they appear in the table.

        Raises
        ------
         TransformerNotFittedError
             If the transformer has not been fitted yet.
        """

    @abstractmethod
    def get_names_of_removed_columns(self) -> list[str]:
        """
        Get the names of all columns that have been removed by the transformer.

        Returns
        -------
        removed_columns:
            A list of names of the removed columns, ordered as they appear in the table the transformer was fitted on.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """

    def fit_and_transform(
        self, table: ExperimentalTable, column_names: list[str] | None = None
    ) -> tuple[Self, ExperimentalTable]:
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
