from typing import Any

from safe_ds.data import SupervisedDataset, Table
from safe_ds.exceptions import LearningError, PredictionError
from sklearn.exceptions import NotFittedError


# noinspection PyProtectedMember
def fit(model: Any, supervised_dataset: SupervisedDataset) -> None:
    """
    Fit a model for a given supervised dataset.

    Parameters
    ----------
    model
        Classifier or Regression from scikit-learn
    supervised_dataset: SupervisedDataset
        the supervised dataset containing the feature and target vectors

    Raises
    ------
    LearningError
        if the supervised dataset contains invalid values or if the training failed
    """
    try:
        model.fit(
            supervised_dataset.feature_vectors._data,
            supervised_dataset.target_values._data,
        )
    except ValueError as exception:
        raise LearningError(str(exception)) from exception
    except Exception as exception:
        raise LearningError(None) from exception


# noinspection PyProtectedMember
def predict(model: Any, dataset: Table) -> Table:
    """
    Predict a target vector using a dataset containing feature vectors. The model has to be trained first

    Parameters
    ----------
    model
        Classifier or Regression from scikit-learn
    dataset: Table
        the dataset containing the feature vectors

    Returns
    -------
    table : Table
        a dataset containing the given feature vectors and the predicted target vector

    Raises
    ------
    PredictionError
        if predicting with the given dataset failed
    """
    try:
        predicted_target_vector = model.predict(dataset._data)
        result_set = dataset._data.copy(deep=True)
        if "target_predictions" in result_set.columns:
            raise ValueError(
                "Dataset already contains 'target_predictions' column. Please rename this column"
            )
        result_set["target_predictions"] = predicted_target_vector
        return Table(result_set)
    except NotFittedError as exception:
        raise PredictionError("The model was not trained") from exception
    except ValueError as exception:
        raise PredictionError(str(exception)) from exception
    except Exception as exception:
        raise PredictionError(None) from exception
