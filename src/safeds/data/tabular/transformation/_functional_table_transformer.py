from __future__ import annotations

from typing import TYPE_CHECKING, Self

from safeds._utils import _structural_hash

from ._table_transformer import TableTransformer

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table

class FunctionalTableTransformer(TableTransformer):
    """
    Learn a transformation for a set of columns in a `Table` and transform another `Table` with the same columns.

    Parameters
    ----------
    column_names:
        The list of columns used to fit the transformer. If `None`, all suitable columns are used.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 func: callable[[Table], Table],
                 ) -> None:
        super().__init__(None)
        self._func: callable[[Table], Table] = func

        

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._func,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """FunctionalTableTransformer is always considered to be fitted."""
        return True

    # ------------------------------------------------------------------------------------------------------------------
    # Learning and transformation
    # ------------------------------------------------------------------------------------------------------------------

    def fit(self, table: Table) -> Self:
        """
        **Note:** For FunctionalTableTransformer this is a no-OP.

        Parameters
        ----------
        table:
            Required only to be consistent with other transformers.

        Returns
        -------
        fitted_transformer:
            self is always fitted.
            
        """
        fitted_transformer = self
        return fitted_transformer

    def transform(self, table: Table) -> Table:
        """
        Apply the callable to a table.

        **Note:** The given table is not modified.

        Parameters
        ----------
        table:
            The table on which on which the callable is executed.

        Returns
        -------
        transformed_table:
            The transformed table.

        Raises
        ------
        #TODO Implement Errors from the table methods
        
        """
        transformed_table = self._func.__call__(table)
        return transformed_table

    def fit_and_transform(self, table: Table) -> tuple[Self, Table]:
        """
        **Note:** For the FunctionalTableTransformer this is the same as transform().

        Parameters
        ----------
        table:
            The table on which the callable .

        Returns
        -------
        fitted_transformer:
            self is always fitted.
        transformed_table:
            The transformed table.
        """
        fitted_transformer = self
        transformed_table = self.transform(table)
        return fitted_transformer, transformed_table
