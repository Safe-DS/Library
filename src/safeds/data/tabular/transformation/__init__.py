"""Classes for transforming tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._discretizer import Discretizer
    from ._experimental_discretizer import ExperimentalDiscretizer
    from ._experimental_imputer import ExperimentalImputer
    from ._experimental_label_encoder import ExperimentalLabelEncoder
    from ._experimental_one_hot_encoder import ExperimentalOneHotEncoder
    from ._experimental_range_scaler import ExperimentalRangeScaler
    from ._experimental_standard_scaler import ExperimentalStandardScaler
    from ._experimental_table_transformer import ExperimentalInvertibleTableTransformer, ExperimentalTableTransformer
    from ._imputer import Imputer
    from ._label_encoder import LabelEncoder
    from ._one_hot_encoder import OneHotEncoder
    from ._range_scaler import RangeScaler
    from ._standard_scaler import StandardScaler
    from ._table_transformer import InvertibleTableTransformer, TableTransformer

apipkg.initpkg(
    __name__,
    {
        "Discretizer": "._discretizer:Discretizer",
        "ExperimentalDiscretizer": "._experimental_discretizer:ExperimentalDiscretizer",
        "ExperimentalImputer": "._experimental_imputer:ExperimentalImputer",
        "ExperimentalLabelEncoder": "._experimental_label_encoder:ExperimentalLabelEncoder",
        "ExperimentalOneHotEncoder": "._experimental_one_hot_encoder:Experimental",
        "ExperimentalRangeScaler": "._experimental_range_scaler:ExperimentalRangeScaler",
        "ExperimentalStandardScaler": "._experimental_standard_scaler:ExperimentalStandardScaler",
        "ExperimentalTableTransformer": "._experimental_table_transformer:ExperimentalTableTransformer",
        "ExperimentalInvertibleTableTransformer": "._experimental_table_transformer:ExperimentalInvertibleTableTransformer",
        "Imputer": "._imputer:Imputer",
        "InvertibleTableTransformer": "._table_transformer:InvertibleTableTransformer",
        "LabelEncoder": "._label_encoder:LabelEncoder",
        "OneHotEncoder": "._one_hot_encoder:OneHotEncoder",
        "RangeScaler": "._range_scaler:RangeScaler",
        "StandardScaler": "._standard_scaler:StandardScaler",
        "TableTransformer": "._table_transformer:TableTransformer",
    },
)

__all__ = [
    "Discretizer",
    "ExperimentalDiscretizer",
    "ExperimentalImputer",
    "ExperimentalLabelEncoder",
    "ExperimentalOneHotEncoder",
    "ExperimentalRangeScaler",
    "ExperimentalStandardScaler",
    "ExperimentalTableTransformer",
    "ExperimentalInvertibleTableTransformer",
    "Imputer",
    "InvertibleTableTransformer",
    "LabelEncoder",
    "OneHotEncoder",
    "RangeScaler",
    "StandardScaler",
    "TableTransformer",
]
