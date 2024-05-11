from __future__ import annotations

import warnings
from typing import Any

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import (
    DatasetMissesDataError,
    DatasetMissesFeaturesError,
    MissingValuesColumnError,
    ModelNotFittedError,
    NonNumericColumnError,
    PredictionError,
)


def _predict_with_sklearn_model(
    model: Any,
    dataset: Table | TabularDataset,
    feature_names: list[str] | None,
    target_name: str | None,
) -> TabularDataset:
    """
    Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

    Parameters
    ----------
    model:
        Classifier or regressor from scikit-learn.
    dataset:
        The dataset containing the features.
    target_name:
        The name of the target column.
    feature_names:
        The names of the feature columns.

    Returns
    -------
    table:
        A dataset containing the given features and the predicted target.

    Raises
    ------
    ModelNotFittedError
        If the model has not been fitted yet.
    DatasetMissesFeaturesError
        If the dataset misses feature columns.
    PredictionError
        If predicting with the given dataset failed.
    NonNumericColumnError
        If the dataset contains non-numerical values.
    MissingValuesColumnError
        If the dataset contains missing values.
    DatasetMissesDataError
        If the dataset contains no rows.
    """
    # Validation
    if model is None or target_name is None or feature_names is None:
        raise ModelNotFittedError
    if isinstance(dataset, TabularDataset):  # pragma: no cover
        dataset = dataset.features

    missing_feature_names = [feature_name for feature_name in feature_names if not dataset.has_column(feature_name)]
    if missing_feature_names:
        raise DatasetMissesFeaturesError(missing_feature_names)

    if dataset.number_of_rows == 0:
        raise DatasetMissesDataError

    non_numerical_column_names_2 = set(dataset.column_names) - set(
        dataset.remove_non_numeric_columns().column_names,
    )
    if len(non_numerical_column_names_2) != 0:
        raise NonNumericColumnError(
            str(non_numerical_column_names_2),
            "You can use the LabelEncoder or OneHotEncoder to transform your non-numerical data to numerical"
            " data.\nThe OneHotEncoder should be used if you work with nominal data. If your data contains too many"
            " different values\nor is ordinal, you should use the LabelEncoder.\n",
        )

    null_containing_column_names_2 = set(dataset.column_names) - set(
        dataset.remove_columns_with_missing_values().column_names,
    )
    if len(null_containing_column_names_2) != 0:
        raise MissingValuesColumnError(
            str(null_containing_column_names_2),
            "You can use the Imputer to replace the missing values based on different strategies.\nIf you want to"
            " remove the missing values entirely you can use the method `Table.remove_rows_with_missing_values`.",
        )

    dataset_df = dataset.remove_columns_except(feature_names)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            predicted_target_vector = model.predict(dataset_df._data_frame)
        output = dataset.remove_columns(target_name).add_columns(
            Column(target_name, predicted_target_vector),
        )

        extra_names = [
            column_name
            for column_name in dataset.column_names
            if column_name != target_name and column_name not in feature_names
        ]

        return TabularDataset(
            output,
            target_name=target_name,
            extra_names=extra_names,
        )
    except ValueError as exception:
        raise PredictionError(str(exception)) from exception
