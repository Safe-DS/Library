from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sklearn.svm import SVR as sk_SVR  # noqa: N811

from safeds.ml.classical._util_sklearn import fit, predict

from safeds.ml.classical.regression._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin

    from safeds.data.tabular.containers import Table, TaggedTable


class SupportVectorMachine(Regressor):
    """
    Support vector machine.

    Parameters
    ----------
    c: float
        The strength of regularization. Must be strictly positive.

    Raises
    ------
    ValueError
        If `c` is less than or equal to 0.
    """

    def __init__(self, *, c: float = 1.0, kernel: SupportVectorMachineKernel = None) -> None:
        # Internal state
        self._wrapped_regressor: sk_SVR | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

        if c <= 0:
            raise ValueError("The parameter 'c' has to be greater than 0.")
        self._c = c
        self._kernel = kernel

    @property
    def c(self) -> float:
        return self._c

    @property
    def kernel(self) -> SupportVectorMachineKernel:
        return self._kernel

    def fit(self, training_set: TaggedTable) -> SupportVectorMachine:
        """
        Create a copy of this regressor and fit it with the given training data.

        This regressor is not modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_regressor : SupportVectorMachine
            The fitted regressor.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        """
        wrapped_regressor = self._get_sklearn_regressor()
        fit(wrapped_regressor, training_set)

        result = SupportVectorMachine(c=self._c)
        result._wrapped_regressor = wrapped_regressor
        result._feature_names = training_set.features.column_names
        result._target_name = training_set.target.name

        return result

    def predict(self, dataset: Table) -> TaggedTable:
        """
        Predict a target vector using a dataset containing feature vectors. The model has to be trained first.

        Parameters
        ----------
        dataset : Table
            The dataset containing the feature vectors.

        Returns
        -------
        table : TaggedTable
            A dataset containing the given feature vectors and the predicted target vector.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        DatasetContainsTargetError
            If the dataset contains the target column already.
        DatasetMissesFeaturesError
            If the dataset misses feature columns.
        PredictionError
            If predicting with the given dataset failed.
        """
        return predict(self._wrapped_regressor, dataset, self._feature_names, self._target_name)

    def is_fitted(self) -> bool:
        """
        Check if the regressor is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the regressor is fitted.
        """
        return self._wrapped_regressor is not None

    def _get_sklearn_regressor(self) -> RegressorMixin:
        """
        Return a new wrapped Regressor from sklearn.

        Returns
        -------
        wrapped_regressor: RegressorMixin
            The sklearn Regressor.
        """
        return sk_SVR(C=self._c)


class SupportVectorMachineKernel(ABC):
    """The abstract base class of the different subclasses supported by the `Kernel`."""

    @abstractmethod
    def get_sklearn_kernel(self, kernel: SupportVectorMachineKernel):
        """
        Get the kernel of the given SupportVectorMachine.

        Parameters
        ----------
        kernel: SupportVectorMachine
        The kernel to get.
        """


class Linear(SupportVectorMachineKernel):
    def get_sklearn_kernel(self):
        return "linear"


class Polynomial(SupportVectorMachineKernel):
    def __init__(self, degree: int):
        if degree < 1:
            raise ValueError("The parameter 'degree' has to be greater than or equal to 1.")
        self._degree = degree

    def get_sklearn_kernel(self):
        return f"poly_{self._degree}"


class Sigmoid(SupportVectorMachineKernel):
    def get_sklearn_kernel(self):
        return "sigmoid"


class RadialBasisFunction(SupportVectorMachineKernel):
    def get_sklearn_kernel(self):
        return "rbf"
