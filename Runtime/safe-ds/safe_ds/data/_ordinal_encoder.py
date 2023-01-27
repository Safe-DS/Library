from safe_ds import exceptions
from safe_ds.data._table import Table
from sklearn import preprocessing


# noinspection PyProtectedMember
class OrdinalEncoder:
    """
    This OrdinalEncoder encodes one or more given columns into ordinal numbers. The encoding order must be provided.

    Parameters
    --------
    order : list[str]
        The order in which the ordinal encoder encodes the values
    """

    def __init__(self, order: list[str]) -> None:
        self.is_fitted = 0
        self.oe = preprocessing.OrdinalEncoder(categories=[order])
        self.order = order

    def fit(self, table: Table, column_name: str) -> None:
        """
        Fit the ordinal encoder with the values in the given table.

        Parameters
        ----------
        table : Table
            The table containing the data to fit the ordinal encoder with.

        column_name : str
            The column which should be ordinal-encoded

        Returns
        -------
        None
            This function does not return any value. It updates the internal state of the ordinal encoder object.

        Raises
        -------
            LearningError if the Model couldn't be fitted correctly
        """

        p_df = table._data
        p_df.columns = table.schema.get_column_names()
        try:
            self.oe.fit(p_df[[column_name]])
        except exceptions.NotFittedError as exc:
            raise exceptions.LearningError("") from exc

    def transform(self, table: Table, column_name: str) -> Table:
        """
        Transform the given table to an ordinal-encoded table.

        Parameters
        ----------
        table:
                table with target values
        column_name:
                name of column as string
        Returns
        -------
        table: Table
            Table with ordinal encodings.

        Raises
        ------
            a NotFittedError if the Model wasn't fitted before transforming
        """
        p_df = table._data.copy()
        p_df.columns = table.schema.get_column_names()
        try:
            p_df[[column_name]] = self.oe.transform(p_df[[column_name]])
            p_df[column_name] = p_df[column_name].astype(dtype="int64", copy=False)
            return Table(p_df)
        except Exception as exc:
            raise exceptions.NotFittedError from exc

    def fit_transform(self, table: Table, columns: list[str]) -> Table:
        """
        Oridnal-encodes a given table with the given ordinal encoder.
        The order is provided in the constructor, a new order will not be inferred from other columns.

        Parameters
        ----------
            table: the table which will be transformed
            columns: list of column names to be considered while encoding

        Returns
        -------
        table: Table
            a new Table object which is ordinal-encoded

        Raises
        -------
            NotFittedError if the encoder wasn't fitted before transforming.
            KeyError if the column doesn`t exist

        """
        try:
            for col in columns:
                # Fit the Ordinal Encoder on the Column
                self.fit(table, col)
                # transform the column using the trained Ordinal Encoder
                table = self.transform(table, col)
            return table
        except exceptions.NotFittedError as exc:
            raise exceptions.NotFittedError from exc

    def inverse_transform(self, table: Table, column_name: str) -> Table:
        """
        Inverse the transformed table back to original encodings.

        Parameters
        ----------
            table:  The table to be inverse transformed.
            column_name: The column which should be ordinal-encoded

        Returns
        -------
        table: Table
            inverse transformed table.

        Raises
        -------
            NotFittedError if the encoder wasn't fitted before transforming.
        """

        p_df = table._data.copy()
        p_df.columns = table.schema.get_column_names()
        try:
            p_df[[column_name]] = self.oe.inverse_transform(p_df[[column_name]])
            return Table(p_df)
        except exceptions.NotFittedError as exc:
            raise exceptions.NotFittedError from exc
