from __future__ import annotations

import warnings
from typing import Any

import pandas
from safeds.data.tabular import Table
from safeds.exceptions import LearningError, NotFittedError
from sklearn import exceptions, preprocessing


def warn(*_: Any, **__: Any) -> None:
    pass


warnings.warn = warn


# noinspection PyProtectedMember


class LabelEncoder:
    """
    The LabelEncoder encodes one or more given columns into labels.
    """

    def __init__(self) -> None:
        self.is_fitted = 0
        self.le = preprocessing.LabelEncoder()

    def fit(self, table: Table, column: str) -> None:
        """
        Fit the label encoder with the values in the table.

        Parameters
        ----------
        table : Table
            The table containing the data used to fit the label encoder.

        column : str
            The list of columns supposed to be label-encoded.

        Returns
        -------
        None
            This function does not return any value. It updates the internal state of the label encoder object.

        Raises
        -------
        LearningError
            If the model fitting was unsuccessful.
        """
        try:

            self.le.fit(table.keep_columns([column])._data)
        except exceptions.NotFittedError as exc:
            raise LearningError("") from exc

    def transform(self, table: Table, column: str) -> Table:
        """
        Transform the given table to a normalized encoded table.

        Parameters
         ----------
         table : Table
                 The table with target values.
         column : str
                 The name of the column.

         Returns
         -------
         result : Table
             Table with normalized encodings.

         Raises
         ------
         NotFittedError
            If the Model wasn't fitted before transforming.
        """
        p_df = table._data
        p_df.columns = table.schema.get_column_names()
        try:
            p_df[column] = self.le.transform(p_df[column])
            return Table(p_df)
        except Exception as exc:
            raise NotFittedError from exc

    def fit_transform(self, table: Table, columns: list[str]) -> Table:
        """
        Label-encode the table with the label encoder.

        Parameters
        ----------
        table : Table
            The table to be transformed.
        columns : list[str]
            The list of column names to be encoded.

        Returns
        -------
        table : Table
            The label-encoded table.

        Raises
        -------
        NotFittedError
            If the encoder wasn't fitted before transforming.

        """
        p_df = table._data
        p_df.columns = table.schema.get_column_names()
        try:
            for col in columns:
                # Fit the LabelEncoder on the Column
                self.le.fit(p_df[col])

                # transform the column using the trained Label Encoder
                p_df[col] = self.le.transform(p_df[col])
            return Table(pandas.DataFrame(p_df))
        except exceptions.NotFittedError as exc:
            raise NotFittedError from exc

    def inverse_transform(self, table: Table, column: str) -> Table:
        """
        Inverse-transform the table back to its original encodings.

        Parameters
        ----------
        table : Table
            The table to be inverse-transformed.
        column : str
            The column to be inverse-transformed.

        Returns
        -------
        table : Table
            The inverse-transformed table.

        Raises
        -------
        NotFittedError
            If the encoder wasn't fitted before transforming.
        """

        try:
            p_df = table._data
            p_df.columns = table.schema.get_column_names()
            p_df[column] = self.le.inverse_transform(p_df[column])
            return Table(p_df)
        except exceptions.NotFittedError as exc:
            raise NotFittedError from exc
