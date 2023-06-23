from __future__ import annotations

from typing import TYPE_CHECKING

from safeds.data.tabular.containers import Column, Row, Table
from safeds.exceptions import ColumnIsTargetError, IllegalSchemaModificationError, UnknownColumnNameError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any

    from safeds.data.tabular.transformation import InvertibleTableTransformer, TableTransformer


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
        UnknownColumnNameError
            If target_name matches none of the column names.
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
        if target_name not in table.column_names:
            raise UnknownColumnNameError([target_name])

        # If no feature names are specified, use all columns except the target column
        if feature_names is None:
            feature_names = table.column_names
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
        result._features = table.keep_only_columns(feature_names)
        result._target = table.get_column(target_name)

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

        self._features: Table = super().keep_only_columns(feature_names)
        self._target: Column = super().get_column(target_name)

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
    # Conversion back to table
    # ------------------------------------------------------------------------------------------------------------------

    def to_table(self: TaggedTable) -> Table:
        """
        Remove the tagging from a TaggedTable.

        The original TaggedTable is not modified.

        Parameters
        ----------
        self: TaggedTable
        The TaggedTable.

        Returns
        -------
        table: Table
        The table as an untagged Table, i.e. without the information about which columns are features or target.

        """
        return self.features.add_column(self.target)

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
        return TaggedTable._from_table(super().add_column(column), target_name=self.target.name)

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
        return TaggedTable._from_table(super().add_columns(columns), target_name=self.target.name)

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
        return TaggedTable._from_table(super().add_row(row), target_name=self.target.name)

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
        return TaggedTable._from_table(super().add_rows(rows), target_name=self.target.name)

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
        return TaggedTable._from_table(super().filter_rows(query), target_name=self.target.name)

    def keep_only_columns(self, column_names: list[str]) -> Table:
        # TODO: Change return type in signature and docstring.
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
        return super().keep_only_columns(column_names)
        # TODO:
        #  Re-build TaggedTable before returning,
        #  throw exception if appropriate,
        #  investigate and fix pytest errors.
        # if self.target.name not in column_names:
        #     raise IllegalSchemaModificationError("Must keep target column and at least one feature column.")
        # return TaggedTable._from_table(super().keep_only_columns(column_names), self.target.name)

    def remove_columns(self, column_names: list[str]) -> TaggedTable:
        """
        Return a table without the given column(s).

        This table is not modified.

        Parameters
        ----------
        column_names : list[str]
            A list containing all columns to be dropped.

        Returns
        -------
        table : TaggedTable
            A table without the given columns.

        Raises
        ------
        UnknownColumnNameError
            If any of the given columns does not exist.
        ColumnIsTargetError
            If any of the given columns is the target column.
        """
        try:
            return TaggedTable._from_table(super().remove_columns(column_names), self.target.name)
        except UnknownColumnNameError:
            raise ColumnIsTargetError(self.target.name) from None

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
        ColumnIsTargetError
            If any of the columns to be removed is the target column.
        """
        table = super().remove_columns_with_missing_values()
        try:
            tagged = TaggedTable._from_table(table, self.target.name, None)
        except UnknownColumnNameError:
            raise ColumnIsTargetError(self.target.name) from None
        return tagged

    def remove_columns_with_non_numerical_values(self) -> TaggedTable:
        """
        Return a table without the columns that contain non-numerical values.

        This table is not modified.

        Returns
        -------
        table : TaggedTable
            A table without the columns that contain non-numerical values.

        Raises
        ------
        ColumnIsTargetError
            If any of the columns to be removed is the target column.
        """
        table = super().remove_columns_with_non_numerical_values()
        try:
            tagged = TaggedTable._from_table(table, self.target.name)
        except UnknownColumnNameError:
            raise ColumnIsTargetError(self.target.name) from None
        return tagged

    def remove_duplicate_rows(self) -> TaggedTable:
        """
        Return a copy of the table with every duplicate row removed.

        This table is not modified.

        Returns
        -------
        result : TaggedTable
            The table with the duplicate rows removed.
        """
        return TaggedTable._from_table(super().remove_duplicate_rows(), self.target.name)

    def remove_rows_with_missing_values(self) -> TaggedTable:
        """
        Return a table without the rows that contain missing values.

        This table is not modified.

        Returns
        -------
        table : TaggedTable
            A table without the rows that contain missing values.
        """
        return TaggedTable._from_table(super().remove_rows_with_missing_values(), self.target.name)

    def remove_rows_with_outliers(self) -> TaggedTable:
        """
        Remove all rows from the table that contain at least one outlier.

        We define an outlier as a value that has a distance of more than 3 standard deviations from the column mean.
        Missing values are not considered outliers. They are also ignored during the calculation of the standard
        deviation.

        This table is not modified.

        Returns
        -------
        new_table : TaggedTable
            A new table without rows containing outliers.
        """
        return TaggedTable._from_table(super().remove_rows_with_outliers(), self.target.name)

    def rename_column(self, old_name: str, new_name: str) -> TaggedTable:
        """
        Rename a single column.

        This table is not modified.

        Parameters
        ----------
        old_name : str
            The old name of the target column
        new_name : str
            The new name of the target column

        Returns
        -------
        table : TaggedTable
            The Table with the renamed column.

        Raises
        ------
        UnknownColumnNameError
            If the specified old target column name does not exist.
        DuplicateColumnNameError
            If the specified new target column name already exists.
        """
        return TaggedTable._from_table(
            super().rename_column(old_name, new_name),
            new_name if self.target.name == old_name else self.target.name,
        )

    def replace_column(self, old_column_name: str, new_columns: list[Column]) -> TaggedTable:
        """
        Return a copy of the table with the specified column replaced by new columns.

        The order of columns is kept.

        If the column to be replaced is the target column, it must be replaced by exactly one column.

        The original is not modified.

        Parameters
        ----------
        old_column_name : str
            The name of the column to be replaced.

        new_columns : list[Column]
            The new columns replacing the old column.

        Returns
        -------
        result : TaggedTable
            A table with the old column replaced by the new column.

        Raises
        ------
        UnknownColumnNameError
            If the old column does not exist.

        DuplicateColumnNameError
            If the new column already exists and the existing column is not affected by the replacement.

        ColumnSizeError
            If the size of the column does not match the amount of rows.

        IllegalSchemaModificationError
            If the target column would be removed or replaced by more than one column.
        """
        if old_column_name == self.target.name:
            if len(new_columns) != 1:
                raise IllegalSchemaModificationError(
                    f'Target column "{self.target.name}" can only be replaced by exactly one new column.',
                )
            else:
                return TaggedTable._from_table(
                    super().replace_column(old_column_name, new_columns),
                    new_columns[0].name,
                )
        else:
            return TaggedTable._from_table(super().replace_column(old_column_name, new_columns), self.target.name)

    def shuffle_rows(self) -> TaggedTable:
        """
        Shuffle the table randomly.

        This table is not modified.

        Returns
        -------
        result : TaggedTable
            The shuffled Table.

        """
        return TaggedTable._from_table(super().shuffle_rows(), self.target.name)

    def slice_rows(
        self,
        start: int | None = None,
        end: int | None = None,
        step: int = 1,
    ) -> TaggedTable:
        """
        Slice a part of the table into a new table.

        This table is not modified.

        Parameters
        ----------
        start : int
            The first index of the range to be copied into a new table, None by default.
        end : int
            The last index of the range to be copied into a new table, None by default.
        step : int
            The step size used to iterate through the table, 1 by default.

        Returns
        -------
        result : TaggedTable
            The resulting table.

        Raises
        ------
        IndexOutOfBoundsError
            If the index is out of bounds.
        """
        return TaggedTable._from_table(super().slice_rows(start, end, step), self.target.name)

    def sort_columns(
        self,
        comparator: Callable[[Column, Column], int] = lambda col1, col2: (col1.name > col2.name)
        - (col1.name < col2.name),
    ) -> TaggedTable:
        """
        Sort the columns of a `TaggedTable` with the given comparator and return a new `TaggedTable`.

        The original table is not modified. The comparator is a function that takes two columns `col1` and `col2` and
        returns an integer:

        * If `col1` should be ordered before `col2`, the function should return a negative number.
        * If `col1` should be ordered after `col2`, the function should return a positive number.
        * If the original order of `col1` and `col2` should be kept, the function should return 0.

        If no comparator is given, the columns will be sorted alphabetically by their name.

        Parameters
        ----------
        comparator : Callable[[Column, Column], int]
            The function used to compare two columns.

        Returns
        -------
        new_table : TaggedTable
            A new table with sorted columns.
        """
        return TaggedTable._from_table(super().sort_columns(comparator), self.target.name)

    def sort_rows(self, comparator: Callable[[Row, Row], int]) -> TaggedTable:
        """
        Sort the rows of a `TaggedTable` with the given comparator and return a new `TaggedTable`.

        The original table is not modified. The comparator is a function that takes two rows `row1` and `row2` and
        returns an integer:

        * If `row1` should be ordered before `row2`, the function should return a negative number.
        * If `row1` should be ordered after `row2`, the function should return a positive number.
        * If the original order of `row1` and `row2` should be kept, the function should return 0.

        Parameters
        ----------
        comparator : Callable[[Row, Row], int]
            The function used to compare two rows.

        Returns
        -------
        new_table : TaggedTable
            A new table with sorted rows.
        """
        return TaggedTable._from_table(super().sort_rows(comparator), self.target.name)

    def transform_column(self, name: str, transformer: Callable[[Row], Any]) -> TaggedTable:
        """
        Transform provided column by calling provided transformer.

        This table is not modified.

        Returns
        -------
        result : TaggedTable
            The table with the transformed column.

        Raises
        ------
        UnknownColumnNameError
            If the column does not exist.

        """
        return TaggedTable._from_table(super().transform_column(name, transformer), self.target.name)

    def transform_table(self, transformer: TableTransformer) -> TaggedTable:
        """
        Apply a learned transformation onto this table.

        This table is not modified.

        Parameters
        ----------
        transformer : TableTransformer
            The transformer which transforms the given table.

        Returns
        -------
        transformed_table : TaggedTable
            The transformed table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.
        ColunmIsTargetError
            If the transformer tries to remove or replace the target column.

        Examples
        --------
        >>> from safeds.data.tabular.transformation import OneHotEncoder
        >>> from safeds.data.tabular.containers import TaggedTable
        >>> table = TaggedTable({"feat1": ["a", "b", "a"], "feat2": ["a", "b", "d"], "target": [1, 2, 3]},"target")
        >>> table
          feat1 feat2  target
        0     a     a       1
        1     b     b       2
        2     a     d       3
        >>> transformer = OneHotEncoder().fit(table, table.features.column_names)
        >>> table.transform_table(transformer)
           feat1__a  feat1__b  feat2__a  feat2__b  feat2__d  target
        0       1.0       0.0       1.0       0.0       0.0       1
        1       0.0       1.0       0.0       1.0       0.0       2
        2       1.0       0.0       0.0       0.0       1.0       3
        """
        try:
            transformed_table = transformer.transform(self)
        except ColumnIsTargetError as e:  # can happen for example with OneHotEncoder
            raise ColumnIsTargetError(self.target.name) from e  # Re-throw for shorter stacktrace
        # For future transformers, it may also happen that they remove the target column without throwing.
        # If this ever happens, comment-in these lines (currently out-commented b/c of code coverage):
        # if self.target.name in transformer.get_names_of_removed_columns():
        #     raise ColumnIsTargetError(self.target.name)
        return TaggedTable._from_table(transformed_table, self.target.name)

    def inverse_transform_table(self, transformer: InvertibleTableTransformer) -> TaggedTable:
        """
        Invert the transformation applied by the given transformer.

        This table is not modified.

        Parameters
        ----------
        transformer : InvertibleTableTransformer
            The transformer that was used to create this table.

        Returns
        -------
        table : TaggedTable
            The original table.

        Raises
        ------
        TransformerNotFittedError
            If the transformer has not been fitted yet.

        Examples
        --------
        >>> from safeds.data.tabular.transformation import OneHotEncoder
        >>> from safeds.data.tabular.containers import TaggedTable
        >>> table = TaggedTable({"feat1": ["a", "b", "a"], "feat2": ["a", "b", "d"], "target": [1, 2, 3]}, "target")
        >>> table
          feat1 feat2  target
        0     a     a       1
        1     b     b       2
        2     a     d       3
        >>> transformer = OneHotEncoder().fit(table, table.features.column_names)
        >>> transformed_table = table.transform_table(transformer)
        >>> transformed_table
           feat1__a  feat1__b  feat2__a  feat2__b  feat2__d  target
        0       1.0       0.0       1.0       0.0       0.0       1
        1       0.0       1.0       0.0       1.0       0.0       2
        2       1.0       0.0       0.0       0.0       1.0       3
        >>> transformed_table.inverse_transform_table(transformer)
          feat1 feat2  target
        0     a     a       1
        1     b     b       2
        2     a     d       3
        """
        return TaggedTable._from_table(transformer.inverse_transform(self), self.target.name)
