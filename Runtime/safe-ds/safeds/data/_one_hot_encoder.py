import pandas as pd
from safeds.data.tabular import Table
from safeds.exceptions import LearningError, NotFittedError
from sklearn import exceptions
from sklearn.preprocessing import OneHotEncoder as OHE_sklearn


class OneHotEncoder:
    """
    The OneHotEncoder encodes categorical columns to numerical features [0,1] that represent the existence for each value.
    """

    def __init__(self) -> None:
        self.encoder = OHE_sklearn()

    def fit(self, table: Table, columns: list[str]) -> None:
        """
        Fit the encoder to a table.

        Parameters
        ----------
        table : Table
            The table used to fit the encoder.
        columns : list[str]:
            The list of columns from the table used to fit the encoder.

        Raises
        ----------
        LearningError
            If there was an error during fitting.
        """
        try:
            table_k_columns = table.keep_columns(column_names=columns)
            df = table_k_columns._data
            df.columns = table_k_columns.schema.get_column_names()
            self.encoder.fit(df)
        except exceptions.NotFittedError as exc:
            raise LearningError("") from exc

    def transform(self, table: Table) -> Table:
        """
        Transform the data with the trained encoder.

        Parameters
        ----------
        table : Table
            The data to be transformed.

        Returns
        ----------
        table : Table
            The transformed table.

        Raises
        ----------
        NotFittedError
            If the encoder wasn't fitted before transforming.
        """
        try:
            table_k_columns = table.keep_columns(self.encoder.feature_names_in_)
            df_k_columns = table_k_columns._data
            df_k_columns.columns = table_k_columns.schema.get_column_names()
            df_new = pd.DataFrame(self.encoder.transform(df_k_columns).toarray())
            df_new.columns = self.encoder.get_feature_names_out()
            df_concat = table._data.copy()
            df_concat.columns = table.schema.get_column_names()
            data_new = pd.concat([df_concat, df_new], axis=1).drop(
                self.encoder.feature_names_in_, axis=1
            )
            return Table(data_new)
        except Exception as exc:
            raise NotFittedError from exc

    def fit_transform(self, table: Table, columns: list[str]) -> Table:
        """
        Fit and transform data with a OneHotEncoder.

        Parameters
        ----------
        table : Table
            The table used to fit the encoder and subsequently to be transformed
        columns : list[str]:
            The list of columns from the table used to fit the encoder and subsequently to be transformed.

        Returns
        ----------
        table : Table
            The transformed table.

        """
        self.fit(table, columns)
        return self.transform(table)

    def inverse_transform(self, table: Table) -> Table:
        """
        Reset a transformed table to its original state.

        Parameters
        ----------
        table : Table
            The table to be inverse-transformed.

        Returns
        ----------
        table : Table
            The inverse-transformed table.

        Raises
        ----------
        NotFittedError
            If the encoder wasn't fitted before transforming.

        """
        try:
            data = self.encoder.inverse_transform(
                table.keep_columns(self.encoder.get_feature_names_out())._data
            )
            df = pd.DataFrame(data)
            df.columns = self.encoder.feature_names_in_
            new_table = Table(df)
            for col in table.drop_columns(
                self.encoder.get_feature_names_out()
            ).to_columns():
                new_table = new_table.add_column(col)
            return new_table
        except exceptions.NotFittedError as exc:
            raise NotFittedError from exc
