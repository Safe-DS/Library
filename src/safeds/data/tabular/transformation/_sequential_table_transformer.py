from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from safeds._utils import _structural_hash
from safeds.exceptions import TransformerNotFittedError, TransformerNotInvertibleError

from ._invertible_table_transformer import InvertibleTableTransformer

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Table

    from ._table_transformer import TableTransformer


class SequentialTableTransformer(InvertibleTableTransformer):
    """
    The SequentialTableTransformer transforms a table using multiple transformers in sequence.

    Parameters
    ----------
    transformers:
        The list of transformers used to transform the table. Used in the order as they are supplied in the list.
    """

    def __init__(
        self,
        transformers: list[TableTransformer],
    ) -> None:
        super().__init__(None)

        # Check if transformers actually contains any transformers.
        if transformers is None or len(transformers) == 0:
            warn(
                "transformers should contain at least 1 transformer",
                UserWarning,
                stacklevel=2,
            )

        # Parameters
        self._transformers: list[TableTransformer] = transformers

        # Internal State
        self._is_fitted: bool = False

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._transformers,
            self._is_fitted,
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer is fitted."""
        return self._is_fitted

    def fit(self, table: Table) -> SequentialTableTransformer:
        """
        Fits all the transformers in order.

        Parameters
        ----------
        table:
            The table used to fit the transformers.

        Returns
        -------
        fitted_transformer:
            The fitted transformer.

        Raises
        ------
        ValueError:
            Raises a ValueError if the table has no rows.
        """
        if table.row_count == 0:
            raise ValueError("The SequentialTableTransformer cannot be fitted because the table contains 0 rows.")

        current_table: Table = table
        fitted_transformers: list[TableTransformer] = []

        for transformer in self._transformers:
            fitted_transformer = transformer.fit(current_table)
            fitted_transformers.append(fitted_transformer)
            current_table = fitted_transformer.transform(current_table)

        result: SequentialTableTransformer = SequentialTableTransformer(
            transformers=fitted_transformers,
        )

        result._is_fitted = True
        return result

    def transform(self, table: Table) -> Table:
        """
        Transform the table using all the transformers sequentially.

        Might change the order and type of columns base on the transformers used.

        Parameters
        ----------
        table:
            The table to be transformed.

        Returns
        -------
        transformed_table:
            The transformed table.

        Raises
        ------
        TransformerNotFittedError:
            Raises a TransformerNotFittedError if the transformer isn't fitted.
        """
        if not self._is_fitted:
            raise TransformerNotFittedError

        current_table: Table = table
        for transformer in self._transformers:
            current_table = transformer.transform(current_table)

        return current_table

    def inverse_transform(self, transformed_table: Table) -> Table:
        """
        Inversely transforms the table using all the transformers sequentially in inverse order.

        Might change the order and type of columns base on the transformers used.

        Parameters
        ----------
        transformed_table:
            The table to be transformed back.

        Returns
        -------
        original_table:
            The original table.

        Raises
        ------
        TransformerNotFittedError:
            Raises a TransformerNotFittedError if the transformer isn't fitted.
        TransformerNotInvertibleError:
            Raises a TransformerNotInvertibleError if one of the transformers isn't invertible.
        """
        if not self._is_fitted:
            raise TransformerNotFittedError

        # sequentially inverse transform the table with all transformers, working from the back of the list forwards.
        current_table: Table = transformed_table
        for transformer in reversed(self._transformers):
            # check if transformer is invertible
            if not (isinstance(transformer, InvertibleTableTransformer)):
                raise TransformerNotInvertibleError(str(type(transformer)))
            current_table = transformer.inverse_transform(current_table)

        return current_table
