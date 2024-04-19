"""Classes for classification tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._ada_boost import AdaBoostClassifier
    from ._classifier import Classifier
    from ._decision_tree import DecisionTreeClassifier
    from ._gradient_boosting import GradientBoostingClassifier
    from ._k_nearest_neighbors import KNearestNeighborsClassifier
    from ._logistic_regression import LogisticRegressionClassifier
    from ._random_forest import RandomForestClassifier
    from ._support_vector_machine import SupportVectorMachineClassifier

apipkg.initpkg(
    __name__,
    {
        "AdaBoostClassifier": "._ada_boost:AdaBoostClassifier",
        "Classifier": "._classifier:Classifier",
        "DecisionTreeClassifier": "._decision_tree:DecisionTreeClassifier",
        "GradientBoostingClassifier": "._gradient_boosting:GradientBoostingClassifier",
        "KNearestNeighborsClassifier": "._k_nearest_neighbors:KNearestNeighborsClassifier",
        "LogisticRegressionClassifier": "._logistic_regression:LogisticRegressionClassifier",
        "RandomForestClassifier": "._random_forest:RandomForestClassifier",
        "SupportVectorMachineClassifier": "._support_vector_machine:SupportVectorMachineClassifier",
    },
)

__all__ = [
    "AdaBoostClassifier",
    "Classifier",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "KNearestNeighborsClassifier",
    "LogisticRegressionClassifier",
    "RandomForestClassifier",
    "SupportVectorMachineClassifier",
]
