from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sklearn.svm import SVC as sk_SVC  # noqa: N811

from safeds.exceptions import ClosedBound, OpenBound, OutOfBoundsError
from safeds.ml.classical._util_sklearn import fit, predict
from safeds.ml.classical.classification import Classifier

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin

    from safeds.data.tabular.containers import Table, TaggedTable


class SupportVectorMachineKernel(ABC):
    """The abstract base class of the different subclasses supported by the `Kernel`."""

    @abstractmethod
    def get_sklearn_kernel(self) -> object:
        """
        Get the kernel of the given SupportVectorMachine.

        Returns
        -------
        object
        The kernel of the SupportVectorMachine.
        """


class SupportVectorMachine(Classifier):
    """
    Support vector machine.

    Parameters
    ----------
    c: float
        The strength of regularization. Must be strictly positive.
    kernel: SupportVectorMachineKernel | None
        The type of kernel to be used. Defaults to None.

    Raises
    ------
    OutOfBoundsError
        If `c` is less than or equal to 0.
    """

    def __init__(self, *, c: float = 1.0, kernel: SupportVectorMachineKernel | None = None) -> None:
        # Internal state
        self._wrapped_classifier: sk_SVC | None = None
        self._feature_names: list[str] | None = None
        self._target_name: str | None = None

        # Hyperparameters
        if c <= 0:
            raise OutOfBoundsError(c, name="c", lower_bound=OpenBound(0))
        self._c = c
        self._kernel = kernel

    @property
    def c(self) -> float:
        """
        Get the regularization strength.

        Returns
        -------
        result: float
            The regularization strength.
        """
        return self._c

    @property
    def kernel(self) -> SupportVectorMachineKernel | None:
        """
        Get the type of kernel used.

        Returns
        -------
        result: SupportVectorMachineKernel | None
            The type of kernel used.
        """
        return self._kernel

    class Kernel:
        class Linear(SupportVectorMachineKernel):
            def get_sklearn_kernel(self) -> str:
                """
                Get the name of the linear kernel.

                Returns
                -------
                result: str
                    The name of the linear kernel.
                """
                return "linear"

        class Polynomial(SupportVectorMachineKernel):
            def __init__(self, degree: int):
                if degree < 1:
                    raise OutOfBoundsError(degree, name="degree", lower_bound=ClosedBound(1))
                self._degree = degree

            def get_sklearn_kernel(self) -> str:
                """
                Get the name of the polynomial kernel.

                Returns
                -------
                result: str
                    The name of the polynomial kernel.
                """
                return "poly"

        class Sigmoid(SupportVectorMachineKernel):
            def get_sklearn_kernel(self) -> str:
                """
                Get the name of the sigmoid kernel.

                Returns
                -------
                result: str
                    The name of the sigmoid kernel.
                """
                return "sigmoid"

        class RadialBasisFunction(SupportVectorMachineKernel):
            def get_sklearn_kernel(self) -> str:
                """
                Get the name of the radial basis function (RBF) kernel.

                Returns
                -------
                result: str
                    The name of the RBF kernel.
                """
                return "rbf"

    def _get_kernel_name(self) -> str:
        """
        Get the name of the kernel.

        Returns
        -------
        result: str
            The name of the kernel.

        Raises
        ------
        TypeError
            If the kernel type is invalid.
        """
        if isinstance(self.kernel, SupportVectorMachine.Kernel.Linear):
            return "linear"
        elif isinstance(self.kernel, SupportVectorMachine.Kernel.Polynomial):
            return "poly"
        elif isinstance(self.kernel, SupportVectorMachine.Kernel.Sigmoid):
            return "sigmoid"
        elif isinstance(self.kernel, SupportVectorMachine.Kernel.RadialBasisFunction):
            return "rbf"
        else:
            raise TypeError("Invalid kernel type.")

    def fit(self, training_set: TaggedTable) -> SupportVectorMachine:
        """
        Create a copy of this classifier and fit it with the given training data.

        This classifier is not modified.

        Parameters
        ----------
        training_set : TaggedTable
            The training data containing the feature and target vectors.

        Returns
        -------
        fitted_classifier : SupportVectorMachine
            The fitted classifier.

        Raises
        ------
        LearningError
            If the training data contains invalid values or if the training failed.
        UntaggedTableError
            If the table is untagged.
        NonNumericColumnError
            If the training data contains non-numerical values.
        MissingValuesColumnError
            If the training data contains missing values.
        DatasetMissesDataError
            If the training data contains no rows.
        """
        wrapped_classifier = self._get_sklearn_classifier()
        fit(wrapped_classifier, training_set)

        result = SupportVectorMachine(c=self._c, kernel=self._kernel)
        result._wrapped_classifier = wrapped_classifier
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
        NonNumericColumnError
            If the dataset contains non-numerical values.
        MissingValuesColumnError
            If the dataset contains missing values.
        DatasetMissesDataError
            If the dataset contains no rows.
        """
        return predict(self._wrapped_classifier, dataset, self._feature_names, self._target_name)

    def is_fitted(self) -> bool:
        """
        Check if the classifier is fitted.

        Returns
        -------
        is_fitted : bool
            Whether the classifier is fitted.
        """
        return self._wrapped_classifier is not None

    def _get_sklearn_classifier(self) -> ClassifierMixin:
        """
        Return a new wrapped Classifier from sklearn.

        Returns
        -------
        wrapped_classifier: ClassifierMixin
            The sklearn Classifier.
        """
        return sk_SVC(C=self._c)
