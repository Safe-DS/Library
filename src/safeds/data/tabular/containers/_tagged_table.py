from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.data.tabular.containers import Column, Table

if TYPE_CHECKING:
    import pandas as pd
    from safeds.data.tabular.typing import Schema


class TaggedTable(Table):
    """
    A tagged table is a table that additionally knows which columns are features and which are the target to predict.

    Parameters
    ----------
    data : Iterable
        The data.
    schema : Schema | None
        The schema of the table. If not specified, the schema will be inferred from the data.
    target_name : str
        Name of the target column.
    feature_names : list[str] | None
        Names of the feature columns. If None, all columns except the target column are used.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _from_table(
        table: Table,
        target_name: str,
        feature_names: list[str] | None = None,
    ) -> TaggedTable:
        """
        Create a tagged table from a table.

        Parameters
        ----------
        table : Table
            The table.
        target_name : str
            Name of the target column.
        feature_names : list[str] | None
            Names of the feature columns. If None, all columns except the target column are used.

        Returns
        -------
        tagged_table : TaggedTable
            The created table.

        Examples
        --------
        >>> from safeds.data.tabular.containers import Table, TaggedTable
        >>> table = Table({"col1": ["a", "b", "c", "a"], "col2": [1, 2, 3, 4]})
        >>> tagged_table = TaggedTable._from_table(table, "col2", ["col1"])
        """
        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = table.column_names
            if target_name in feature_names:
                feature_names.remove(target_name)

        # Validate inputs
        if target_name in feature_names:
            raise ValueError(f"Column '{target_name}' cannot be both feature and target.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must be specified.")

        # Create result
        result = object.__new__(TaggedTable)

        result._data = table._data
        result._schema = table.schema
        result._features = result.keep_only_columns(feature_names)
        result._target = result.get_column(target_name)

        return result

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    # noinspection PyMissingConstructor
    def __init__(
        self,
        data: pd.DataFrame,
        schema: Schema,
        target_name: str,
        feature_names: list[str] | None = None,
    ):
        self._data = data
        self._schema = schema

        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = self.column_names
            if target_name in feature_names:
                feature_names.remove(target_name)

        # Validate inputs
        if target_name in feature_names:
            raise ValueError(f"Column '{target_name}' cannot be both feature and target.")
        if len(feature_names) == 0:
            raise ValueError("At least one feature column must be specified.")

        self._features: Table = self.keep_only_columns(feature_names)
        self._target: Column = self.get_column(target_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def features(self) -> Table:
        return self._features

    @property
    def target(self) -> Column:
        return self._target
