from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ._experimental_table_transformer import ExperimentalTableTransformer

if TYPE_CHECKING:
    from safeds.data.tabular.containers import ExperimentalTable


class ExperimentalInvertibleTableTransformer(ExperimentalTableTransformer):
    """A `TableTransformer` that can also undo the learned transformation after it has been applied."""

    @abstractmethod
    def inverse_transform(self, transformed_table: ExperimentalTable) -> ExperimentalTable:
        """
        Undo the learned transformation.

        The table is not modified.

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
