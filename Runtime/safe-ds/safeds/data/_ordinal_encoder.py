from safeds import exceptions
from safeds.data.tabular import Table
from sklearn import preprocessing


# noinspection PyProtectedMember
class OrdinalEncoder:
    """
    This OrdinalEncoder encodes one or more given columns into ordinal numbers. The encoding order must be provided.

    Parameters
    --------
    order : list[str]
        The order in which the ordinal encoder encodes the values.
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
            The table containing the data used to fit the ordinal encoder.

        column_name : str
            The column which should be ordinal-encoded.

        Returns
        -------
        None
            This function does not return any value. It updates the internal state of the ordinal encoder object.

        Raises
        -------
        LearningError
            If the model could not be fitted correctly.
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
        table : Table
            The table with target values.
        column_name : str
            The name of the column.
        Returns
        -------
        table : Table
            The table with ordinal encodings.

        Raises
        ------
        NotFittedError
            If the model was not fitted before transforming.
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
        Oridnal-encode a given table with the given ordinal encoder.
        The order is provided in the constructor. A new order will not be inferred from other columns.

        Parameters
        ----------
        table : Table
            The table which will be transformed.
        columns : list[str]
            The list of column names to be considered while encoding.

        Returns
        -------
        table : Table
            A new Table object which is ordinal-encoded.

        Raises
        -------
        NotFittedError
            If the encoder was not fitted before transforming.
        KeyError
            If the column does not exist.

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
        table : Table
            The table to be inverse-transformed.
        column_name : str
            The column which should be inverse-transformed.

        Returns
        -------
        table : Table
            The inverse-transformed table.

        Raises
        -------
        NotFittedError
            If the encoder was not fitted before transforming.
        """

        p_df = table._data.copy()
        p_df.columns = table.schema.get_column_names()
        try:
            p_df[[column_name]] = self.oe.inverse_transform(p_df[[column_name]])
            return Table(p_df)
        except exceptions.NotFittedError as exc:
            raise exceptions.NotFittedError from exc
