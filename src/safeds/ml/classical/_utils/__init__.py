"""Utilities to work with classical machine learning models."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._fit_sklearn_model import _fit_sklearn_model_in_place
    from ._predict_with_sklearn_model import _predict_with_sklearn_model

apipkg.initpkg(
    __name__,
    {
        "_fit_sklearn_model": "._fit_sklearn_model:_fit_sklearn_model",
        "_predict_with_sklearn_model": "._predict_with_sklearn_model:_predict_with_sklearn_model",
    },
)

__all__ = [
    "_fit_sklearn_model_in_place",
    "_predict_with_sklearn_model",
]
