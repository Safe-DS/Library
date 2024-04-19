from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor

from safeds._config import _get_device
from safeds.data.image.containers import ImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.tabular.containers import Table
from safeds.exceptions import NonNumericColumnError, OutputLengthMismatchError, IndexOutOfBoundsError


class ImageDataset:

    def __init__(self, input_data: ImageList, output_data: ImageList | Table, batch_size=1, shuffle=False) -> None:
        self._shuffle_tensor_indices = torch.LongTensor(list(range(len(input_data))))
        self._shuffle_after_epoch = shuffle
        self._batch_size = batch_size
        self._next_batch_index = 0

        if not isinstance(input_data, _SingleSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")
        else:
            self._input = input_data
        if (isinstance(output_data, Table) and len(input_data) != output_data.number_of_rows) or (isinstance(output_data, ImageList) and len(input_data) != len(output_data)):
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
        elif isinstance(output_data, _SingleSizeImageList):
            _output = output_data.clone()._as_single_size_image_list()
        else:
            raise ValueError("The given output ImageList contains images of different sizes.")
        self._output = _output

    def _get_batch(self, batch_number: int, batch_size: int | None = None) -> tuple[Tensor, Tensor]:
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

    def __iter__(self) -> ImageDataset:
        # self._batch_index = 0
        # if self._shuffle_after_epoch:
        #     self._shuffle_inplace()
        # return self

        # def _generator():
        #     batch_index = 0
        #
        #     while batch_index * self._batch_size < len(self._input):
        #         yield self._get_batch(batch_index)
        #
        #         batch_index += 1
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

    def shuffle(self) -> ImageDataset:
        im_dataset: ImageDataset = copy.copy(self)
        im_dataset._shuffle_tensor_indices = torch.randperm(len(self))
        im_dataset._next_batch_index = 0
        return im_dataset

    # def _shuffle_inplace(self) -> None:
    #     self._shuffle_tensor_indices = torch.randperm(len(self))
    #
    # def _reset_indices_inplace(self) -> None:
    #     self._shuffle_tensor_indices = torch.LongTensor(list(range(len(self))))


class _TableAsTensor:

    def __init__(self, table: Table) -> None:
        self._column_names = table.column_names

        columns_as_tensors = []
        for column_name in table.column_names:
            columns_as_tensors.append(torch.Tensor(table.get_column(column_name)._data.values.astype(np.float32)).unsqueeze(dim=0))

        self._tensor = torch.cat(columns_as_tensors, dim=0).to(_get_device()).T

        if not torch.all(self._tensor.sum(dim=1) == torch.ones(self._tensor.size(dim=0))):
            raise ValueError("The given table is not correctly one hot encoded as it contains rows that have a sum not equal to 1.")
