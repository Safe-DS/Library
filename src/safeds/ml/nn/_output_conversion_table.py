from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Column, Table
from safeds.ml.nn import OutputConversion


class OutputConversionTable(OutputConversion[Table, TabularDataset]):
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

    def _data_conversion(self, input_data: Table, output_data: Tensor, **kwargs: Any) -> TabularDataset:  # noqa: ARG002
        return input_data.add_columns([Column(self._prediction_name, output_data.tolist())]).to_tabular_dataset(
            self._prediction_name,
        )
