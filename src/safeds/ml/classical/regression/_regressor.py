from __future__ import annotations

from abc import ABC
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, wait
from typing import TYPE_CHECKING, Self

from joblib._multiprocessing_helpers import mp

from safeds.data.labeled.containers import TabularDataset
from safeds.exceptions import (
    ColumnLengthMismatchError,
    DatasetMissesDataError,
    ModelNotFittedError,
)
from safeds.ml.classical import SupervisedModel
from safeds.ml.metrics import RegressionMetrics, RegressorMetric

if TYPE_CHECKING:
    from safeds.data.tabular.containers import Column, Table


class Regressor(SupervisedModel, ABC):
    """A model for regression tasks."""

    # ------------------------------------------------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------------------------------------------------

    def summarize_metrics(self, validation_or_test_set: Table | TabularDataset) -> Table:
        """
        Summarize the regressor's metrics on the given data.

        **Note:** The model must be fitted.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        metrics:
            A table containing the regressor's metrics.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return RegressionMetrics.summarize(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
        )

    def coefficient_of_determination(self, validation_or_test_set: Table | TabularDataset) -> float:
        """
        Compute the coefficient of determination (R²) of the regressor on the given data.

        The coefficient of determination compares the regressor's predictions to another model that always predicts the
        mean of the target values. It is a measure of how well the regressor explains the variance in the target values.

        The **higher** the coefficient of determination, the better the regressor. Results range from negative infinity
        to 1.0. You can interpret the coefficient of determination as follows:

        | R²         | Interpretation                                                                             |
        | ---------- | ------------------------------------------------------------------------------------------ |
        | 1.0        | The model perfectly predicts the target values. Did you overfit?                           |
        | (0.0, 1.0) | The model is better than predicting the mean of the target values. You should be here.     |
        | 0.0        | The model is as good as predicting the mean of the target values. Try something else.      |
        | (-∞, 0.0)  | The model is worse than predicting the mean of the target values. Something is very wrong. |

        **Notes:**

        - The model must be fitted.
        - Some other libraries call this metric `r2_score`.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        coefficient_of_determination:
            The coefficient of determination of the regressor.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return RegressionMetrics.coefficient_of_determination(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
        )

    def mean_absolute_error(self, validation_or_test_set: Table | TabularDataset) -> float:
        """
        Compute the mean absolute error (MAE) of the regressor on the given data.

        The mean absolute error is the average of the absolute differences between the predicted and expected target
        values. The **lower** the mean absolute error, the better the regressor. Results range from 0.0 to positive
        infinity.

        **Note:** The model must be fitted.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        mean_absolute_error:
            The mean absolute error of the regressor.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return RegressionMetrics.mean_absolute_error(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
        )

    def mean_directional_accuracy(self, validation_or_test_set: Table | TabularDataset) -> float:
        """
        Compute the mean directional accuracy (MDA) of the regressor on the given data.

        This metric compares two consecutive target values and checks if the predicted direction (down/unchanged/up)
        matches the expected direction. The mean directional accuracy is the proportion of correctly predicted
        directions. The **higher** the mean directional accuracy, the better the regressor. Results range from 0.0 to
        1.0.

        This metric is useful for time series data, where the order of the target values has a meaning. It is not useful
        for other types of data. Because of this, it is not included in the `summarize_metrics` method.

        **Note:** The model must be fitted.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        mean_directional_accuracy:
            The mean directional accuracy of the regressor.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return RegressionMetrics.mean_directional_accuracy(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
        )

    def mean_squared_error(self, validation_or_test_set: Table | TabularDataset) -> float:
        """
        Compute the mean squared error (MSE) of the regressor on the given data.

        The mean squared error is the average of the squared differences between the predicted and expected target
        values. The **lower** the mean squared error, the better the regressor. Results range from 0.0 to positive
        infinity.

        **NoteS:**

        - The model must be fitted.
        - To get the root mean squared error (RMSE), take the square root of the result.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        mean_squared_error:
            The mean squared error of the regressor.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return RegressionMetrics.mean_squared_error(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
        )

    def median_absolute_deviation(self, validation_or_test_set: Table | TabularDataset) -> float:
        """
        Compute the median absolute deviation (MAD) of the regressor on the given data.

        The median absolute deviation is the median of the absolute differences between the predicted and expected
        target values. The **lower** the median absolute deviation, the better the regressor. Results range from 0.0 to
        positive infinity.

        **Note:** The model must be fitted.

        Parameters
        ----------
        validation_or_test_set:
            The validation or test set.

        Returns
        -------
        median_absolute_deviation:
            The median absolute deviation of the regressor.

        Raises
        ------
        ModelNotFittedError
            If the classifier has not been fitted yet.
        """
        if not self.is_fitted:
            raise ModelNotFittedError

        validation_or_test_set = _extract_table(validation_or_test_set)

        return RegressionMetrics.median_absolute_deviation(
            self.predict(validation_or_test_set),
            validation_or_test_set.get_column(self.get_target_name()),
        )

    def fit_by_exhaustive_search(self, training_set: TabularDataset, optimization_metric: RegressorMetric) -> Self:
        """
        Use the hyperparameter choices to create multiple models and fit them.

        **Note:** This model is not modified.

        Parameters
        ----------
        training_set:
            The training data containing the features and target.
        optimization_metric:
            The metric that should be used for determining the performance of a model.

        Returns
        -------
        best_model:
            The model that performed the best out of all possible models given the Choices of hyperparameters.

        Raises
        ------
        PlainTableError
            If a table is passed instead of a TabularDataset.
        DatasetMissesDataError
            If the given training set contains no data.
        FittingWithoutChoiceError
            When trying to call this method on a model without hyperparameter choices.
        LearningError
            If the training data contains invalid values or if the training failed.
        """
        if training_set.to_table().row_count == 0:
            raise DatasetMissesDataError

        self._check_additional_fit_by_exhaustive_search_preconditions()

        [train_split, test_split] = training_set.to_table().split_rows(0.75)
        train_data = train_split.to_tabular_dataset(
            target_name=training_set.target.name,
            extra_names=training_set.extras.column_names,
        )
        test_data = test_split.to_tabular_dataset(
            target_name=training_set.target.name,
            extra_names=training_set.extras.column_names,
        )

        list_of_models = self._get_models_for_all_choices()
        list_of_fitted_models = []

        with ProcessPoolExecutor(max_workers=len(list_of_models), mp_context=mp.get_context("spawn")) as executor:
            futures = []
            for model in list_of_models:
                futures.append(executor.submit(model.fit, train_data))
            [done, _] = wait(futures, return_when=ALL_COMPLETED)
            for future in done:
                list_of_fitted_models.append(future.result())
        executor.shutdown()

        best_model = None
        best_metric_value = None
        for fitted_model in list_of_fitted_models:
            if best_model is None:
                best_model = fitted_model
                match optimization_metric.value:
                    case "mean_squared_error":
                        best_metric_value = fitted_model.mean_squared_error(test_data)
                    case "mean_absolute_error":
                        best_metric_value = fitted_model.mean_absolute_error(test_data)
                    case "median_absolute_deviation":
                        best_metric_value = fitted_model.median_absolute_deviation(test_data)
                    case "coefficient_of_determination":
                        best_metric_value = fitted_model.coefficient_of_determination(test_data)
            else:
                match optimization_metric.value:
                    case "mean_squared_error":
                        error_of_fitted_model = fitted_model.mean_squared_error(test_data)
                        if error_of_fitted_model < best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "mean_absolute_error":
                        error_of_fitted_model = fitted_model.mean_absolute_error(test_data)
                        if error_of_fitted_model < best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "median_absolute_deviation":
                        error_of_fitted_model = fitted_model.median_absolute_deviation(test_data)
                        if error_of_fitted_model < best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
                    case "coefficient_of_determination":
                        error_of_fitted_model = fitted_model.coefficient_of_determination(test_data)
                        if error_of_fitted_model > best_metric_value:
                            best_model = fitted_model  # pragma: no cover
                            best_metric_value = error_of_fitted_model  # pragma: no cover
        assert best_model is not None
        return best_model


def _check_metrics_preconditions(actual: Column, expected: Column) -> None:  # pragma: no cover
    if not actual.type.is_numeric:
        raise TypeError(f"Column 'actual' is not numerical but {actual.type}.")
    if not expected.type.is_numeric:
        raise TypeError(f"Column 'expected' is not numerical but {expected.type}.")

    if actual.row_count != expected.row_count:
        raise ColumnLengthMismatchError(
            "\n".join(
                [
                    f"{actual.name}: {actual.row_count}",
                    f"{expected.name}: {expected.row_count}",
                ],
            ),
        )


def _extract_table(table_or_dataset: Table | TabularDataset) -> Table:
    """Extract the table from the given table or dataset."""
    if isinstance(table_or_dataset, TabularDataset):
        return table_or_dataset.to_table()
    else:
        return table_or_dataset
