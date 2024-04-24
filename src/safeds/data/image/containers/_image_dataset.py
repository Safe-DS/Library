from __future__ import annotations

import copy
from typing import TYPE_CHECKING, TypeVar, Generic

from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.image.typing import ImageSize
from safeds.data.tabular.containers import Table, Column
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import NonNumericColumnError, OutputLengthMismatchError, IndexOutOfBoundsError, \
    TransformerNotFittedError

if TYPE_CHECKING:
    from torch import Tensor

T = TypeVar("T", Column, Table, ImageList)


class ImageDataset(Generic[T]):

    def __init__(self, input_data: ImageList, output_data: T, batch_size=1, shuffle=False) -> None:
        import torch

        self._shuffle_tensor_indices = torch.LongTensor(list(range(len(input_data))))
        self._shuffle_after_epoch = shuffle
        self._batch_size = batch_size
        self._next_batch_index = 0

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")
        else:
            self._input_size = ImageSize(input_data.widths[0], input_data.heights[0], input_data.channel)
            self._input = input_data
        if ((isinstance(output_data, Table) or isinstance(output_data, Column)) and len(input_data) != output_data.number_of_rows) or (isinstance(output_data, ImageList) and len(input_data) != len(output_data)):
            raise OutputLengthMismatchError(f"{len(input_data)} != {output_data.number_of_rows if isinstance(output_data, Table) else len(output_data)}")
        if isinstance(output_data, Table):
            non_numerical_columns = []
            wrong_interval_columns = []
            for column_name in output_data.column_names:
                if not output_data.get_column_type(column_name).is_numeric():
                    non_numerical_columns.append(column_name)
                elif output_data.get_column(column_name).minimum() < 0 or output_data.get_column(column_name).maximum() > 1:
                    wrong_interval_columns.append(column_name)
            if len(non_numerical_columns) > 0:
                raise NonNumericColumnError(f"Columns {non_numerical_columns} are not numerical.")
            if len(wrong_interval_columns) > 0:
                raise ValueError(f"Columns {wrong_interval_columns} have values outside of the interval [0, 1].")
            _output = _TableAsTensor(output_data)
            self._output_size = output_data.number_of_columns
        elif isinstance(output_data, Column):
            _output = _ColumnAsTensor(output_data)
            self._output_size = len(_output._one_hot_encoder.get_names_of_added_columns())
        elif isinstance(output_data, _SingleSizeImageList):
            _output = output_data.clone()._as_single_size_image_list()
            self._output_size = ImageSize(output_data.widths[0], output_data.heights[0], output_data.channel)
        else:
            raise ValueError("The given output ImageList contains images of different sizes.")
        self._output = _output

    def __iter__(self) -> ImageDataset:
        if self._shuffle_after_epoch:
            im_ds = self.shuffle()
        else:
            im_ds = copy.copy(self)
        im_ds._next_batch_index = 0
        return im_ds

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self._next_batch_index * self._batch_size >= len(self._input):
            raise StopIteration
        self._next_batch_index += 1
        return self._get_batch(self._next_batch_index - 1)

    def __len__(self) -> int:
        return self._input.number_of_images

    @property
    def input_size(self) -> ImageSize:
        return self._input_size

    @property
    def output_size(self) -> ImageSize | int:
        return self._output_size

    def get_input(self) -> ImageList:
        return self._input

    def get_output(self) -> T:
        output = self._output
        if isinstance(output, _TableAsTensor):
            return output._to_table()
        elif isinstance(output, _ColumnAsTensor):
            return output._to_column()
        else:
            return output

    def _get_batch(self, batch_number: int, batch_size: int | None = None) -> tuple[Tensor, Tensor]:
        import torch
        from torch import Tensor

        if batch_size is None:
            batch_size = self._batch_size
        if batch_size * batch_number >= len(self._input):
            raise IndexOutOfBoundsError(batch_size * batch_number)
        max_index = batch_size * (batch_number + 1) if batch_size * (batch_number + 1) < len(self._input) else len(self._input)
        input_tensor = self._input._tensor[self._shuffle_tensor_indices[[self._input._indices_to_tensor_positions[index] for index in range(batch_size * batch_number, max_index)]]].to(torch.float32) / 255
        output_tensor: Tensor
        if isinstance(self._output, _SingleSizeImageList):
            output_tensor = self._output._tensor[self._shuffle_tensor_indices[[self._output._indices_to_tensor_positions[index] for index in range(batch_size * batch_number, max_index)]]].to(torch.float32) / 255
        else:  # _output is instance of _TableAsTensor
            output_tensor = self._output._tensor[self._shuffle_tensor_indices[batch_size * batch_number:max_index]]
        return input_tensor, output_tensor

    def shuffle(self) -> ImageDataset[T]:
        import torch
        im_dataset: ImageDataset[T] = copy.copy(self)
        im_dataset._shuffle_tensor_indices = torch.randperm(len(self))
        im_dataset._next_batch_index = 0
        return im_dataset


class _TableAsTensor:

    def __init__(self, table: Table) -> None:
        import torch

        self._column_names = table.column_names
        self._tensor = torch.Tensor(table._data.to_numpy(copy=True)).to(torch.get_default_device())

        if not torch.all(self._tensor.sum(dim=1) == torch.ones(self._tensor.size(dim=0))):
            raise ValueError("The given table is not correctly one hot encoded as it contains rows that have a sum not equal to 1.")

    @staticmethod
    def _from_tensor(tensor: Tensor, column_names: list[str]) -> _TableAsTensor:
        if tensor.dim() != 2:
            raise ValueError(f"Tensor has an invalid amount of dimensions. Needed 2 dimensions but got {tensor.dim()}.")
        if tensor.size(dim=1) != len(column_names):
            raise ValueError(f"Tensor and column_names have different amounts of classes ({tensor.size(dim=1)}!={column_names}.")
        table_as_tensor = _TableAsTensor.__new__(_TableAsTensor)
        table_as_tensor._tensor = tensor
        table_as_tensor._column_names = column_names
        return table_as_tensor

    def _to_table(self) -> Table:
        table = Table(dict(zip(self._column_names, self._tensor.T.tolist())))
        return table


class _ColumnAsTensor:

    def __init__(self, column: Column) -> None:
        import torch

        self._column_name = column.name
        column_as_table = Table.from_columns([column])
        self._one_hot_encoder = OneHotEncoder().fit(column_as_table, [self._column_name])
        self._tensor = torch.Tensor(self._one_hot_encoder.transform(column_as_table)._data.to_numpy(copy=True)).to(torch.get_default_device())

    @staticmethod
    def _from_tensor(tensor: Tensor, column_name: str, one_hot_encoder: OneHotEncoder) -> _ColumnAsTensor:
        if tensor.dim() != 2:
            raise ValueError(f"Tensor has an invalid amount of dimensions. Needed 2 dimensions but got {tensor.dim()}.")
        if not one_hot_encoder.is_fitted():
            raise TransformerNotFittedError()
        if tensor.size(dim=1) != len(one_hot_encoder.get_names_of_added_columns()):
            raise ValueError(f"Tensor and one_hot_encoder have different amounts of classes ({tensor.size(dim=1)}!={one_hot_encoder.get_names_of_added_columns()}.")
        table_as_tensor = _ColumnAsTensor.__new__(_ColumnAsTensor)
        table_as_tensor._tensor = tensor
        table_as_tensor._column_name = column_name
        table_as_tensor._one_hot_encoder = one_hot_encoder
        return table_as_tensor

    def _to_column(self) -> Column:
        table = Table(dict(zip(self._one_hot_encoder.get_names_of_added_columns(), self._tensor.T.tolist())))
        return self._one_hot_encoder.inverse_transform(table).get_column(self._column_name)
