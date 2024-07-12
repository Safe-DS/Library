from __future__ import annotations

from typing import TYPE_CHECKING

from safeds._utils import _structural_hash

if TYPE_CHECKING:
    from collections.abc import Callable

    from safeds.data.tabular.containers import Table

from ._table_transformer import TableTransformer


class FunctionalTableTransformer(TableTransformer):
    """
    Wraps a callable so that it conforms to the TableTransformer interface.

    Parameters
    ----------
    transformer:
        A callable that receives a table and returns a table.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        transformer: Callable[[Table], Table],
    ) -> None:
        super().__init__(None)
        self._transformer = transformer

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._transformer,
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

    def fit(self, table: Table) -> FunctionalTableTransformer:  # noqa: ARG002
        """
        **Note:** For FunctionalTableTransformer this is a no-OP.

        Parameters
        ----------
        table:
            Required only to be consistent with other transformers.

        Returns
        -------
        fitted_transformer:
            Returns self, because this transformer is always fitted.
        """
        return self

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
        Exception:
            Raised when the wrapped callable encounters an error.
        """
        return self._transformer(table)

    def fit_and_transform(self, table: Table) -> tuple[FunctionalTableTransformer, Table]:
        """
        **Note:** For the FunctionalTableTransformer this is the same as transform().

        Parameters
        ----------
        table:
            The table on which the callable is to be executed.

        Returns
        -------
        fitted_transformer:
            Return self because the transformer is always fitted.
        transformed_table:
            The transformed table.
        """
        fitted_transformer = self
        transformed_table = self.transform(table)
        return fitted_transformer, transformed_table
