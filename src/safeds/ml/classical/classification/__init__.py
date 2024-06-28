"""Classes for classification tasks."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._ada_boost_classifier import AdaBoostClassifier
    from ._baseline_classifier import BaselineClassifier
    from ._classifier import Classifier
    from ._decision_tree_classifier import DecisionTreeClassifier
    from ._gradient_boosting_classifier import GradientBoostingClassifier
    from ._k_nearest_neighbors_classifier import KNearestNeighborsClassifier
    from ._logistic_classifier import LogisticClassifier
    from ._random_forest_classifier import RandomForestClassifier
    from ._support_vector_classifier import SupportVectorClassifier

apipkg.initpkg(
    __name__,
    {
        "AdaBoostClassifier": "._ada_boost_classifier:AdaBoostClassifier",
        "BaselineClassifier": "._baseline_classifier:BaselineClassifier",
        "Classifier": "._classifier:Classifier",
        "DecisionTreeClassifier": "._decision_tree_classifier:DecisionTreeClassifier",
        "GradientBoostingClassifier": "._gradient_boosting_classifier:GradientBoostingClassifier",
        "KNearestNeighborsClassifier": "._k_nearest_neighbors_classifier:KNearestNeighborsClassifier",
        "LogisticClassifier": "._logistic_classifier:LogisticClassifier",
        "RandomForestClassifier": "._random_forest_classifier:RandomForestClassifier",
        "SupportVectorClassifier": "._support_vector_classifier:SupportVectorClassifier",
    },
)

__all__ = [
    "AdaBoostClassifier",
    "BaselineClassifier",
    "Classifier",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "KNearestNeighborsClassifier",
    "LogisticClassifier",
    "RandomForestClassifier",
    "SupportVectorClassifier",
]
