from torch import Tensor

from safeds.data.tabular.containers import TaggedTable, Table, Column
from safeds.ml.nn._output_conversion import _OutputConversion


class OutputConversionTable(_OutputConversion[Table, TaggedTable]):
    """
    The output conversion for a neural network, defines the output parameters for the neural network.
    """

    def __init__(self, prediction_name: str = "prediction") -> None:
        self._prediction_name = prediction_name

    def _data_conversion(self, input_data: Table, output_data: Tensor) -> TaggedTable:
        return input_data.add_column(Column(self._prediction_name, output_data.tolist())).tag_columns(self._prediction_name)
