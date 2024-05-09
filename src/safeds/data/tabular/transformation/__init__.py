"""Classes for transforming tabular data."""

from typing import TYPE_CHECKING

import apipkg

if TYPE_CHECKING:
    from ._discretizer import Discretizer
    from ._experimental_discretizer import ExperimentalDiscretizer
    from ._experimental_invertible_table_transformer import ExperimentalInvertibleTableTransformer
    from ._experimental_label_encoder import ExperimentalLabelEncoder
    from ._experimental_one_hot_encoder import ExperimentalOneHotEncoder
    from ._experimental_range_scaler import ExperimentalRangeScaler
    from ._experimental_simple_imputer import ExperimentalSimpleImputer
    from ._experimental_standard_scaler import ExperimentalStandardScaler
    from ._experimental_table_transformer import ExperimentalTableTransformer
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
        "ExperimentalInvertibleTableTransformer": "._experimental_invertible_table_transformer:ExperimentalInvertibleTableTransformer",
        "ExperimentalLabelEncoder": "._experimental_label_encoder:ExperimentalLabelEncoder",
        "ExperimentalOneHotEncoder": "._experimental_one_hot_encoder:ExperimentalOneHotEncoder",
        "ExperimentalRangeScaler": "._experimental_range_scaler:ExperimentalRangeScaler",
        "ExperimentalSimpleImputer": "._experimental_simple_imputer:ExperimentalSimpleImputer",
        "ExperimentalStandardScaler": "._experimental_standard_scaler:ExperimentalStandardScaler",
        "ExperimentalTableTransformer": "._experimental_table_transformer:ExperimentalTableTransformer",
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
    "ExperimentalInvertibleTableTransformer",
    "ExperimentalLabelEncoder",
    "ExperimentalOneHotEncoder",
    "ExperimentalRangeScaler",
    "ExperimentalSimpleImputer",
    "ExperimentalStandardScaler",
    "ExperimentalTableTransformer",
    "Imputer",
    "InvertibleTableTransformer",
    "LabelEncoder",
    "OneHotEncoder",
    "RangeScaler",
    "StandardScaler",
    "TableTransformer",
]
