from __future__ import annotations

from typing import Any

from safeds._utils import _structural_hash
from safeds.data.tabular.containers import Table
from ._table_transformer import TableTransformer
from safeds.exceptions import TransformerNotFittedError, SafeDsError

from  ._invertible_table_transformer import InvertibleTableTransformer

class SequentialTableTransformer(InvertibleTableTransformer):
    """
    The SequentialTableTransforrmer transforms a table using multiple trnasformers in sequence.

    Parameters
    ----------
    transformers:
        The list of transformers used to transform the table. Used in the order as they are supplied in the list.
    
    Raises
    ------
    ValueError:
        Raises a ValueError if the list of Transformers is None or contains no transformers.
    """

    def _init_(
        self, 
        *, 
        transformers: list[TableTransformer]#,
        #column_names: str | list[str] | None = None
    ) -> None:
        super().__init__(None)

        #Check if transformers actually contains any transformers.
        if transformers == None or len(transformers) == 0:
            raise ValueError("transformers must contain at least 1 transformer")

        # Parameters
        self._transformers: list[TableTransformer] = transformers

        # Internal State
        self._is_fitted: bool = False

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._transformers,
            self._is_fitted
        )

    def fit(self, table: Table) -> SequentialTableTransformer:
        """
        Fits all of the transformers in order.

        Parameters
        ----------
        table: 
            The table used to fit the transformers.
        
        Returns
        -------
        The fitted transformer.

        Raises
        ------
        ValueError:
            Raises a ValueError if the table has no rows.
        """
        if table.row_count == 0:
            raise ValueError("The SequentialTable cannot be fitted because the table contains 0 rows")

        current_table: Table = table
        result: SequentialTableTransformer = SequentialTableTransformer(
            transformer=self._transformers, column_names=self._column_names)

        for transformer in result._transformers:
            transformer = transformer.fit(current_table)
            current_table = transformer.transform(current_table)
        
        result._is_fitted = True
        return result
    
    def transform(self, table:Table) -> Table:
        """
        Transforms the table using all the transformers sequentially.

        Parameters
        ----------
        table: 
            The table to be transformed.
        
        Returns
        -------
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
    
    def inverse_transform(self, transformed_table:Table) -> Table:
        #TODO: Replace SafeDsError with a custom error.
        """
        Inversely transforms the table using all the transformers sequentially in inverse order.

        Parameters
        ----------
        table: 
            The table to be transformed back.
        
        Returns
        -------
        The untranformed table.

        Raises
        ------
        TransformerNotFittedError:
            Raises a TransformerNotFittedError if the transformer isn't fitted.
        SafeDsError:
            Raises a SafeDsError if one of the transformers isn't invertable.
        """
        if not self._is_fitted:
            raise TransformerNotFittedError
        
        #check if transformer is invertable
        for transformer in self._transformers:
            if not (hasattr(transformer, "inverse_transform") and callable(getattr(transformer, "inverse_transform"))):
                raise SafeDsError(str(type(transformer)) + " is not invertable!")

        #sequentially inverse transform the table with all transformers, working from the back of the list forwards.
        current_table: Table = transformed_table
        for transformer in reversed(self._transformers):
            current_table = transformer.inverse_transform(current_table)
            
        return current_table