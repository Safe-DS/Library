from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.data.tabular.containers import Column, Row, Table
from safeds.exceptions import ColumnIsTaggedError, UnknownColumnNameError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any


class TaggedTable(Table):
    """
    A tagged table is a table that additionally knows which columns are features and which are the target to predict.

    Parameters
    ----------
    data : Mapping[str, Sequence[Any]]
        The data.
    target_name : str
        Name of the target column.
    feature_names : list[str] | None
        Names of the feature columns. If None, all columns except the target column are used.

    Raises
    ------
    ColumnLengthMismatchError
        If columns have different lengths.
    ValueError
        If the target column is also a feature column.
    ValueError
        If no feature columns are specified.

    Examples
    --------
    >>> from safeds.data.tabular.containers import Table, TaggedTable
    >>> table = Table({"col1": ["a", "b"], "col2": [1, 2]})
    >>> tagged_table = table.tag_columns("col2", ["col1"])
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

        Raises
        ------
        ValueError
            If the target column is also a feature column.
        ValueError
            If no feature columns are specified.

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

    def __init__(
        self,
        data: Mapping[str, Sequence[Any]],
        target_name: str,
        feature_names: list[str] | None = None,
    ):
        """
        Create a tagged table from a mapping of column names to their values.

        Parameters
        ----------
        data : Mapping[str, Sequence[Any]]
            The data.
        target_name : str
            Name of the target column.
        feature_names : list[str] | None
            Names of the feature columns. If None, all columns except the target column are used.

        Raises
        ------
        ColumnLengthMismatchError
            If columns have different lengths.
        ValueError
            If the target column is also a feature column.
        ValueError
            If no feature columns are specified.

        Examples
        --------
        >>> from safeds.data.tabular.containers import TaggedTable
        >>> table = TaggedTable({"a": [1, 2, 3], "b": [4, 5, 6]}, "b", ["a"])
        """
        super().__init__(data)

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

    # ------------------------------------------------------------------------------------------------------------------
    # Overriden methods from Table class:
    # ------------------------------------------------------------------------------------------------------------------

    def add_column(self, column: Column) -> TaggedTable:
        """
        Return the original table with the provided column attached at the end, as a feature column.

        This table is not modified.

        Returns
        -------
        result : TaggedTable
            The table with the column attached as a feature column.

        Raises
        ------
        DuplicateColumnNameError
            If the new column already exists.

        ColumnSizeError
            If the size of the column does not match the amount of rows.

        """
        return TaggedTable._from_table(super().add_column(column), target_name=self.target.name, feature_names=None)

    def add_columns(self, columns: list[Column] | Table) -> TaggedTable:
        """
        Add multiple columns to the table, as feature columns.

        This table is not modified.

        Parameters
        ----------
        columns : list[Column] or Table
            The columns to be added.

        Returns
        -------
        result: TaggedTable
            A new table combining the original table and the given columns as feature columns.

        Raises
        ------
        ColumnSizeError
            If at least one of the column sizes from the provided column list does not match the table.
        DuplicateColumnNameError
            If at least one column name from the provided column list already exists in the table.
        """
        return TaggedTable._from_table(super().add_columns(columns), target_name=self.target.name, feature_names=None)

    def add_row(self, row: Row) -> TaggedTable:
        """
        Add a row to the table.

        This table is not modified.

        Parameters
        ----------
        row : Row
            The row to be added.

        Returns
        -------
        table : TaggedTable
            A new table with the added row at the end.

        Raises
        ------
        SchemaMismatchError
            If the schema of the row does not match the table schema.
        """
        return TaggedTable._from_table(super().add_row(row), target_name=self.target.name, feature_names=None)

    def add_rows(self, rows: list[Row] | Table) -> TaggedTable:
        """
        Add multiple rows to a table.

        This table is not modified.

        Parameters
        ----------
        rows : list[Row] or Table
            The rows to be added.

        Returns
        -------
        result : TaggedTable
            A new table which combines the original table and the given rows.

        Raises
        ------
        SchemaMismatchError
            If the schema of on of the row does not match the table schema.
        """
        return TaggedTable._from_table(super().add_rows(rows), target_name=self.target.name, feature_names=None)

    def filter_rows(self, query: Callable[[Row], bool]) -> TaggedTable:
        """
        Return a table with rows filtered by Callable (e.g. lambda function).

        This table is not modified.

        Parameters
        ----------
        query : lambda function
            A Callable that is applied to all rows.

        Returns
        -------
        table : TaggedTable
            A table containing only the rows filtered by the query.
        """
        return TaggedTable._from_table(super().filter_rows(query), target_name=self.target.name, feature_names=None)

    def keep_only_columns(self, column_names: list[str]) -> Table:
        """
        Return a table with only the given column(s).

        This table is not modified.

        Parameters
        ----------
        column_names : list[str]
            A list containing only the columns to be kept.

        Returns
        -------
        table : Table
            A table containing only the given column(s).

        Raises
        ------
        UnknownColumnNameError
            If any of the given columns does not exist.
        IllegalSchemaModificationError
            If none of the given columns is the target column.
        """
        # TODO: Change return type to TaggedTable (2x in docstring, 1x in function definition),
        #  re-build TaggedTable before returning,
        #  throw exception if appropriate,
        #  investigate and fix pytest errors
        # if self.target.name not in column_names:
        # raise IllegalSchemaModificationError(f'Must keep target column "{self.target.name}".')
        return super().keep_only_columns(column_names)

    def remove_columns(self, column_names: list[str]) -> Table:
        # TODO: Change return type to TaggedTable (in function definition and in docstring).
        """
        Return a table without the given column(s).

        This table is not modified.

        Parameters
        ----------
        column_names : list[str]
            A list containing all columns to be dropped.

        Returns
        -------
        table : Table
            A table without the given columns.

        Raises
        ------
        UnknownColumnNameError
            If any of the given columns does not exist.
        ColumnIsTaggedError
            If any of the given columns is the target column.
        """
        try:
            return TaggedTable._from_table(super().remove_columns(column_names), self.target.name, None)
        except UnknownColumnNameError:
            # TODO: Don't return; throw exception and handle it correctly in tests.
            # raise ColumnIsTaggedError({self.target.name})
            return super().remove_columns(column_names)

    def remove_columns_with_missing_values(self) -> TaggedTable:
        """
        Return a table without the columns that contain missing values.

        This table is not modified.

        Returns
        -------
        table : TaggedTable
            A table without the columns that contain missing values.

        Raises
        ------
        ColumnIsTaggedError
            If any of the columns to be removed is the target column.
        """
        table = super().remove_columns_with_missing_values()
        try:
            tagged = TaggedTable._from_table(table, self.target.name, None)
        except UnknownColumnNameError:
            raise ColumnIsTaggedError(self.target.name) from None
        return tagged
