import pandas as pd
from safe_ds.data import Table
from safe_ds.exceptions import LearningError, NotFittedError
from sklearn import exceptions
from sklearn.preprocessing import OneHotEncoder as OHE_sklearn


class OneHotEncoder:
    """
    This OneHotEncoder encodes columns in numerical columns that represent the existance for each value [0,1]
    """

    def __init__(self) -> None:
        self.encoder = OHE_sklearn()

    def fit(self, table: Table, columns: list[str]) -> None:
        """
        Fit the encoder to the data

        Parameters
        ----------
        table: Table
            the data to fit
        columns: list[str]:
            a list of columns you want to fit
        """
        try:
            self.encoder.fit(table.keep_columns(column_names=columns)._data)
        except exceptions.NotFittedError as exc:
            raise LearningError("") from exc

    def transform(self, table: Table) -> Table:
        """
        Transform the data with the trained encoder

        Parameters
        ----------
        table: Table
            the data to transform

        Returns
        ----------
        table: Table
            the transformed table
        """
        try:
            df_new = pd.DataFrame(
                self.encoder.transform(
                    table.keep_columns(self.encoder.feature_names_in_)._data.copy()
                ).toarray()
            )
            df_new.columns = self.encoder.get_feature_names_out()
            data_new = pd.concat([table._data.copy(), df_new], axis=1).drop(
                self.encoder.feature_names_in_, axis=1
            )
            return Table(data_new)
        except Exception as exc:
            raise NotFittedError from exc

    def fit_transform(self, table: Table, columns: list[str]) -> Table:
        """
        Fit and transform the data with a OneHotEncoder

        Parameters
        ----------
        table: Table
            the data you want to fit and transform
        columns: list[str]:
            a list of columns you want to transform and fit

        Returns
        ----------
        table: Table
            the transformed table
        """
        self.fit(table, columns)
        return self.transform(table)

    def inverse_transform(self, table: Table) -> Table:
        """
        Reset a transformed Table to its original state

        Parameters
        ----------
        table: Table
            the Table you want to reset

        Returns
        ----------
        table: Table
            the resettet table
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
