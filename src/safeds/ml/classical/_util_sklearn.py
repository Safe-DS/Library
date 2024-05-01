import warnings
from typing import Any

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    DatasetMissesDataError,
    DatasetMissesFeaturesError,
    LearningError,
    MissingValuesColumnError,
    ModelNotFittedError,
    NonNumericColumnError,
    PlainTableError,
    PredictionError,
)


# noinspection PyProtectedMember
def fit(model: Any, tabular_dataset: TabularDataset) -> None:
    """
    Fit a model for a given tabular dataset.

    Parameters
    ----------
    model:
        Classifier or Regression from scikit-learn.
    tabular_dataset:
        The tabular dataset containing the feature and target vectors.

    Raises
    ------
    LearningError
        If the tabular dataset contains invalid values or if the training failed.
    TypeError
        If a table is passed instead of a tabular dataset.
    NonNumericColumnError
        If the training data contains non-numerical values.
    MissingValuesColumnError
        If the training data contains missing values.
    DatasetMissesDataError
        If the training data contains no rows.
    """
    if not isinstance(tabular_dataset, TabularDataset) and isinstance(tabular_dataset, Table):
        raise PlainTableError

    if tabular_dataset._table.number_of_rows == 0:
        raise DatasetMissesDataError

    non_numerical_column_names = set(tabular_dataset.features.column_names) - set(
        tabular_dataset.features.remove_columns_with_non_numerical_values().column_names,
    )
    if len(non_numerical_column_names) != 0:
        raise NonNumericColumnError(
            str(non_numerical_column_names),
            "You can use the LabelEncoder or OneHotEncoder to transform your non-numerical data to numerical"
            " data.\nThe OneHotEncoder should be used if you work with nominal data. If your data contains too many"
            " different values\nor is ordinal, you should use the LabelEncoder.",
        )

    null_containing_column_names = set(tabular_dataset.features.column_names) - set(
        tabular_dataset.features.remove_columns_with_missing_values().column_names,
    )
    if len(null_containing_column_names) != 0:
        raise MissingValuesColumnError(
            str(null_containing_column_names),
            "You can use the Imputer to replace the missing values based on different strategies.\nIf you want to"
            " remove the missing values entirely you can use the method `Table.remove_rows_with_missing_values`.",
        )

    try:
        model.fit(
            tabular_dataset.features._data,
            tabular_dataset.target._data,
        )
    except ValueError as exception:
        raise LearningError(str(exception)) from exception


# noinspection PyProtectedMember
def predict(model: Any, dataset: Table, feature_names: list[str] | None, target_name: str | None) -> TabularDataset:
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
    missing_feature_names = [feature_name for feature_name in feature_names if not dataset.has_column(feature_name)]
    if missing_feature_names:
        raise DatasetMissesFeaturesError(missing_feature_names)
    if isinstance(dataset, TabularDataset):
        dataset = dataset.features  # Cast to Table type, so Python will call the right methods...

    if dataset.number_of_rows == 0:
        raise DatasetMissesDataError

    non_numerical_column_names = set(dataset.keep_only_columns(feature_names).column_names) - set(
        dataset.keep_only_columns(feature_names).remove_columns_with_non_numerical_values().column_names,
    )
    if len(non_numerical_column_names) != 0:
        raise NonNumericColumnError(
            str(non_numerical_column_names),
            "You can use the LabelEncoder or OneHotEncoder to transform your non-numerical data to numerical"
            " data.\nThe OneHotEncoder should be used if you work with nominal data. If your data contains too many"
            " different values\nor is ordinal, you should use the LabelEncoder.\n",
        )

    null_containing_column_names = set(dataset.keep_only_columns(feature_names).column_names) - set(
        dataset.keep_only_columns(feature_names).remove_columns_with_missing_values().column_names,
    )
    if len(null_containing_column_names) != 0:
        raise MissingValuesColumnError(
            str(null_containing_column_names),
            "You can use the Imputer to replace the missing values based on different strategies.\nIf you want to"
            " remove the missing values entirely you can use the method `Table.remove_rows_with_missing_values`.",
        )

    dataset_df = dataset.keep_only_columns(feature_names)._data
    dataset_df.columns = feature_names

    result_set = dataset._data.reset_index(drop=True)
    result_set.columns = dataset.column_names

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            predicted_target_vector = model.predict(dataset_df.values)
        result_set[target_name] = predicted_target_vector

        extra_names = [
            column_name
            for column_name in dataset.column_names
            if column_name != target_name and column_name not in feature_names
        ]

        return Table._from_pandas_dataframe(result_set).to_tabular_dataset(
            target_name=target_name,
            extra_names=extra_names,
        )
    except ValueError as exception:
        raise PredictionError(str(exception)) from exception
