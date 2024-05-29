from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ._table_transformer import TableTransformer

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table


class InvertibleTableTransformer(TableTransformer, ABC):
    """A `TableTransformer` that can also undo the learned transformation after it has been applied."""

    @abstractmethod
    def inverse_transform(self, transformed_table: Table) -> Table:
        """
        Undo the learned transformation as well as possible.

        Column order and types may differ from the original table. Likewise, some values might not be restored.

        **Note:** The given table is not modified.

        Parameters
        ----------
        transformed_table:
            The table to be transformed back to the original version.

        Returns
        -------
        original_table:
            The original table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        """
