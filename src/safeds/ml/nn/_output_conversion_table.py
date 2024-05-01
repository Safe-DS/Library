from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from safeds.data.labeled.containers import TaggedTable
from safeds.data.tabular.containers import Column, Table
from safeds.ml.nn._output_conversion import _OutputConversion


class OutputConversionTable(_OutputConversion[Table, TaggedTable]):
    """The output conversion for a neural network, defines the output parameters for the neural network."""

    def __init__(self, prediction_name: str = "prediction") -> None:
        """
        Define the output parameters for the neural network in the output conversion.

        Parameters
        ----------
        prediction_name:
            The name of the new column where the prediction will be stored.
        """
        self._prediction_name = prediction_name

    def _data_conversion(self, input_data: Table, output_data: Tensor) -> TaggedTable:
        return input_data.add_column(Column(self._prediction_name, output_data.tolist())).tag_columns(
            self._prediction_name,
        )
