"""Classes to evaluate the performance of machine learning models."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._classification_metrics import ClassificationMetrics
    from ._regression_metrics import RegressionMetrics

apipkg.initpkg(
    __name__,
    {
        "ClassificationMetrics": "._classification_metrics:ClassificationMetrics",
        "RegressionMetrics": "._regression_metrics:RegressionMetrics",
    },
)

__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
]
