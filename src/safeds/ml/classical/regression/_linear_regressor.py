from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sklearn.linear_model import ElasticNet as SklearnElasticNet
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.linear_model import LinearRegression as SklearnLinear
from sklearn.linear_model import Ridge as SklearnRidge

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound
from safeds.exceptions import FittingWithChoiceError, FittingWithoutChoiceError
from safeds.ml.hyperparameters import Choice

from ._regressor import Regressor

if TYPE_CHECKING:
    from sklearn.base import RegressorMixin


class LinearRegressor(Regressor):
    """
    Linear regression.

    Parameters
    ----------
    penalty:
        The type of penalty to be used. Defaults to a simple linear regression.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Inner classes
    # ------------------------------------------------------------------------------------------------------------------

    class Penalty(ABC):
        """
        Possible penalties for the linear regressor.

        Use the static factory methods to create instances of this class.
        """

        @abstractmethod
        def __eq__(self, other: object) -> bool: ...

        @abstractmethod
        def __hash__(self) -> int: ...

        @abstractmethod
        def __str__(self) -> str: ...

        @abstractmethod
        def _get_sklearn_model(self) -> RegressorMixin:
            """Get the model of a penalty."""

        @abstractmethod
        def _get_models_for_all_choices(self) -> list[LinearRegressor]:
            """Get a list of all possible models, given the choices."""

        @abstractmethod
        def _contains_choice_parameters(self) -> bool:
            """Return if any parameters of this penalty are choice instances."""

        @staticmethod
        def linear() -> LinearRegressor.Penalty:
            """Create a linear penalty."""
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def ridge(alpha: float | Choice[float] = 1.0) -> LinearRegressor.Penalty:
            """Create a ridge penalty."""
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def lasso(alpha: float | Choice[float] = 1.0) -> LinearRegressor.Penalty:
            """Create a lasso penalty."""
            raise NotImplementedError  # pragma: no cover

        @staticmethod
        def elastic_net(
            alpha: float | Choice[float] = 1.0,
            lasso_ratio: float | Choice[float] = 0.5,
        ) -> LinearRegressor.Penalty:
            """Create an elastic net penalty."""
            raise NotImplementedError  # pragma: no cover

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, penalty: LinearRegressor.Penalty | None | Choice[LinearRegressor.Penalty | None] = None) -> None:
        Regressor.__init__(self)
        if penalty is None:
            penalty = LinearRegressor.Penalty.linear()

        # Hyperparameters
        self._penalty: LinearRegressor.Penalty | Choice[LinearRegressor.Penalty | None] = penalty

    def __hash__(self) -> int:
        return _structural_hash(
            super().__hash__(),
            self._penalty,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def penalty(self) -> LinearRegressor.Penalty | Choice[LinearRegressor.Penalty | None]:
        """The regularization of the model."""
        return self._penalty

    def _clone(self) -> LinearRegressor:
        return LinearRegressor(penalty=self._penalty)

    def _get_sklearn_model(self) -> RegressorMixin:
        assert not isinstance(self.penalty, Choice)
        return self.penalty._get_sklearn_model()

    def _check_additional_fit_preconditions(self) -> None:
        if isinstance(self._penalty, Choice) or self.penalty._contains_choice_parameters():  # type: ignore[union-attr]
            raise FittingWithChoiceError

    def _check_additional_fit_by_exhaustive_search_preconditions(self) -> None:
        if not isinstance(self._penalty, Choice) and not self.penalty._contains_choice_parameters():  # type: ignore[union-attr]
            raise FittingWithoutChoiceError

    def _get_models_for_all_choices(self) -> list[LinearRegressor]:
        penalty_choices = self._penalty if isinstance(self._penalty, Choice) else [self._penalty]

        models = []
        for pen in penalty_choices:
            if pen is None:
                models.append(LinearRegressor())
            elif pen._contains_choice_parameters():
                models.extend(pen._get_models_for_all_choices())
            else:
                models.append(LinearRegressor(penalty=pen))
        return models


# ----------------------------------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------------------------------


class _Linear(LinearRegressor.Penalty):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Linear):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__qualname__,
        )

    def __str__(self) -> str:
        return "Linear"

    def _contains_choice_parameters(self) -> bool:
        return False

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _get_sklearn_model(self) -> SklearnLinear:
        return SklearnLinear(n_jobs=-1)

    def _get_models_for_all_choices(self) -> list[LinearRegressor]:
        raise NotImplementedError  # pragma: no cover


class _Ridge(LinearRegressor.Penalty):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, alpha: float | Choice[float] = 1.0):
        # Validation
        if isinstance(alpha, Choice):
            for a in alpha:
                _check_bounds("alpha", a, lower_bound=_ClosedBound(0))
        else:
            _check_bounds("alpha", alpha, lower_bound=_ClosedBound(0))

        self._alpha = alpha

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Ridge):
            return NotImplemented
        return self._alpha == other._alpha

    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__qualname__,
            self._alpha,
        )

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._alpha)

    def __str__(self) -> str:
        return f"Ridge(alpha={self._alpha})"

    def _contains_choice_parameters(self) -> bool:
        return isinstance(self._alpha, Choice)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def alpha(self) -> float | Choice[float]:
        """The regularization of the linear penalty."""
        return self._alpha

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _get_sklearn_model(self) -> SklearnRidge:
        return SklearnRidge(alpha=self._alpha)

    def _get_models_for_all_choices(self) -> list[LinearRegressor]:
        assert isinstance(self._alpha, Choice)
        models = []
        for alpha in self._alpha:
            models.append(LinearRegressor(penalty=LinearRegressor.Penalty.ridge(alpha=alpha)))
        return models


class _Lasso(LinearRegressor.Penalty):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, alpha: float | Choice[float] = 1.0):
        # Validation
        if isinstance(alpha, Choice):
            for a in alpha:
                _check_bounds("alpha", a, lower_bound=_ClosedBound(0))
        else:
            _check_bounds("alpha", alpha, lower_bound=_ClosedBound(0))

        self._alpha = alpha

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Lasso):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__qualname__,
            self._alpha,
        )

    def __str__(self) -> str:
        return f"Lasso(alpha={self._alpha})"

    def _contains_choice_parameters(self) -> bool:
        return isinstance(self._alpha, Choice)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def alpha(self) -> float | Choice[float]:
        """The regularization of the linear penalty."""
        return self._alpha

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _get_sklearn_model(self) -> SklearnLasso:
        return SklearnLasso(alpha=self._alpha)

    def _get_models_for_all_choices(self) -> list[LinearRegressor]:
        assert isinstance(self._alpha, Choice)
        models = []
        for alpha in self._alpha:
            models.append(LinearRegressor(penalty=LinearRegressor.Penalty.lasso(alpha=alpha)))
        return models


class _ElasticNet(LinearRegressor.Penalty):
    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, alpha: float | Choice[float] = 1.0, lasso_ratio: float | Choice[float] = 0.5):
        # Validation
        if isinstance(alpha, Choice):
            for a in alpha:
                _check_bounds("alpha", a, lower_bound=_ClosedBound(0))
        else:
            _check_bounds("alpha", alpha, lower_bound=_ClosedBound(0))

        if isinstance(lasso_ratio, Choice):
            for lr in lasso_ratio:
                _check_bounds("lasso_ratio", lr, lower_bound=_ClosedBound(0), upper_bound=_ClosedBound(1))
        else:
            _check_bounds("lasso_ratio", lasso_ratio, lower_bound=_ClosedBound(0), upper_bound=_ClosedBound(1))

        self._alpha = alpha
        self._lasso_ratio = lasso_ratio

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ElasticNet):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return _structural_hash(
            self.__class__.__qualname__,
            self._alpha,
            self._lasso_ratio,
        )

    def __str__(self) -> str:
        return f"ElasticNet(alpha={self._alpha}, lasso_ratio={self._lasso_ratio})"

    def _contains_choice_parameters(self) -> bool:
        return isinstance(self._alpha, Choice) or isinstance(self._lasso_ratio, Choice)

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def alpha(self) -> float | Choice[float]:
        """The regularization of the linear penalty."""
        return self._alpha

    @property
    def lasso_ratio(self) -> float | Choice[float]:
        """The regularization of the linear penalty."""
        return self._lasso_ratio

    # ------------------------------------------------------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------------------------------------------------------

    def _get_sklearn_model(self) -> SklearnElasticNet:
        return SklearnElasticNet(alpha=self._alpha, l1_ratio=self._lasso_ratio)

    def _get_models_for_all_choices(self) -> list[LinearRegressor]:
        alpha_choices = self._alpha if isinstance(self._alpha, Choice) else [self._alpha]
        lasso_choices = self._lasso_ratio if isinstance(self._lasso_ratio, Choice) else [self._lasso_ratio]

        models = []
        for alpha in alpha_choices:
            for lasso in lasso_choices:
                models.append(
                    LinearRegressor(penalty=LinearRegressor.Penalty.elastic_net(alpha=alpha, lasso_ratio=lasso)),
                )
        return models


# Override the methods with classes, so they can be used in `isinstance` calls. Unlike methods, classes define a type.
# This is needed for the DSL, where LinearRegressor penalties are variants of an enum.
LinearRegressor.Penalty.linear = _Linear  # type: ignore[method-assign]
LinearRegressor.Penalty.ridge = _Ridge  # type: ignore[method-assign]
LinearRegressor.Penalty.lasso = _Lasso  # type: ignore[method-assign]
LinearRegressor.Penalty.elastic_net = _ElasticNet  # type: ignore[method-assign]
