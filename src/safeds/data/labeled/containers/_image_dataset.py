from __future__ import annotations

import copy
import sys
import warnings
from typing import TYPE_CHECKING, Generic, TypeVar

from safeds._config import _get_device, _init_default_device
from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound
from safeds.data.image.containers import ImageList
from safeds.data.image.containers._empty_image_list import _EmptyImageList
from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.image.typing import ImageSize
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.transformation import OneHotEncoder
from safeds.exceptions import (
    IndexOutOfBoundsError,
    NonNumericColumnError,
    OutputLengthMismatchError,
    TransformerNotFittedError,
)

from ._dataset import Dataset

if TYPE_CHECKING:
    from torch import Tensor

T = TypeVar("T", Column, Table, ImageList)


class ImageDataset(Generic[T], Dataset):
    """
    A Dataset for ImageLists as input and ImageLists, Tables or Columns as output.

    Parameters
    ----------
    input_data:
        the input ImageList
    output_data:
        the output data
    batch_size:
        the batch size used for training
    shuffle:
        weather the data should be shuffled after each epoch of training
    """

    def __init__(self, input_data: ImageList, output_data: T, batch_size: int = 1, shuffle: bool = False) -> None:
        import torch

        _init_default_device()

        self._shuffle_tensor_indices: torch.LongTensor = torch.LongTensor(list(range(len(input_data))))
        self._shuffle_after_epoch: bool = shuffle
        self._batch_size: int = batch_size
        self._next_batch_index: int = 0

        if isinstance(input_data, _MultiSizeImageList):
            raise ValueError("The given input ImageList contains images of different sizes.")  # noqa: TRY004
        elif isinstance(input_data, _EmptyImageList):
            raise ValueError("The given input ImageList contains no images.")  # noqa: TRY004
        else:
            self._input_size: ImageSize = ImageSize(input_data.widths[0], input_data.heights[0], input_data.channel)
            self._input: _SingleSizeImageList = input_data._as_single_size_image_list()
        if ((isinstance(output_data, Column | Table)) and len(input_data) != output_data.number_of_rows) or (
            isinstance(output_data, ImageList) and len(input_data) != len(output_data)
        ):
            if isinstance(output_data, Table):
                output_len = output_data.number_of_rows
            else:
                output_len = len(output_data)
            raise OutputLengthMismatchError(f"{len(input_data)} != {output_len}")
        if isinstance(output_data, Table):
            non_numerical_columns = []
            wrong_interval_columns = []
            for column_name in output_data.column_names:
                if not output_data.get_column_type(column_name).is_numeric:
                    non_numerical_columns.append(column_name)
                elif (output_data.get_column(column_name).min() or 0) < 0 or (
                    output_data.get_column(column_name).max() or 0
                ) > 1:
                    wrong_interval_columns.append(column_name)
            if len(non_numerical_columns) > 0:
                raise NonNumericColumnError(f"Columns {non_numerical_columns} are not numerical.")
            if len(wrong_interval_columns) > 0:
                raise ValueError(f"Columns {wrong_interval_columns} have values outside of the interval [0, 1].")
            _output: _TableAsTensor | _ColumnAsTensor | _SingleSizeImageList = _TableAsTensor(output_data)
            _output_size: int | ImageSize = output_data.number_of_columns
        elif isinstance(output_data, Column):
            _column_as_tensor = _ColumnAsTensor(output_data)
            _output_size = len(_column_as_tensor._one_hot_encoder._get_names_of_added_columns())
            _output = _column_as_tensor
        elif isinstance(output_data, _SingleSizeImageList):
            _output = output_data._clone()._as_single_size_image_list()
            _output_size = ImageSize(output_data.widths[0], output_data.heights[0], output_data.channel)
        else:
            raise ValueError("The given output ImageList contains images of different sizes.")  # noqa: TRY004
        self._output = _output  # type: ignore[var-annotated]  # TODO: check what the type should be
        self._output_size = _output_size  # type: ignore[var-annotated]  # TODO: check what the type should be

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

    def __eq__(self, other: object) -> bool:
        """
        Compare two image datasets.

        Parameters
        ----------
        other:
            The image dataset to compare to.

        Returns
        -------
        equals:
            Whether the two image datasets are the same.
        """
        if not isinstance(other, ImageDataset):
            return NotImplemented
        return (self is other) or (
            self._shuffle_after_epoch == other._shuffle_after_epoch
            and self._batch_size == other._batch_size
            and isinstance(other._output, type(self._output))
            and (self._input == other._input)
            and (self._output == other._output)
        )

    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this image dataset.

        Returns
        -------
        hash:
            the hash value
        """
        return _structural_hash(self._input, self._output, self._shuffle_after_epoch, self._batch_size)

    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """
        return (
            sys.getsizeof(self._shuffle_tensor_indices)
            + self._shuffle_tensor_indices.element_size() * self._shuffle_tensor_indices.nelement()
            + sys.getsizeof(self._input)
            + sys.getsizeof(self._output)
            + sys.getsizeof(self._input_size)
            + sys.getsizeof(self._output_size)
            + sys.getsizeof(self._shuffle_after_epoch)
            + sys.getsizeof(self._batch_size)
            + sys.getsizeof(self._next_batch_index)
        )

    @property
    def input_size(self) -> ImageSize:
        """
        Get the input `ImageSize` of this dataset.

        Returns
        -------
        input_size:
            the input `ImageSize`
        """
        return self._input_size

    @property
    def output_size(self) -> ImageSize | int:
        """
        Get the output size of this dataset.

        Returns
        -------
        output_size:
            the output size
        """
        return self._output_size

    def get_input(self) -> ImageList:
        """
        Get the input data of this dataset.

        Returns
        -------
        input:
            the input data of this dataset
        """
        return self._input

    def get_output(self) -> T:
        """
        Get the output data of this dataset.

        Returns
        -------
        output:
            the output data of this dataset
        """
        output = self._output
        if isinstance(output, _TableAsTensor):
            return output._to_table()  # type: ignore[return-value]
        elif isinstance(output, _ColumnAsTensor):
            return output._to_column()  # type: ignore[return-value]
        else:
            return output  # type: ignore[return-value]

    def _get_batch(self, batch_number: int, batch_size: int | None = None) -> tuple[Tensor, Tensor]:
        import torch

        _init_default_device()

        if batch_size is None:
            batch_size = self._batch_size

        _check_bounds("batch_size", batch_size, lower_bound=_ClosedBound(1))

        if batch_number < 0 or batch_size * batch_number >= len(self._input):
            raise IndexOutOfBoundsError(batch_size * batch_number)
        max_index = (
            batch_size * (batch_number + 1) if batch_size * (batch_number + 1) < len(self._input) else len(self._input)
        )
        input_tensor = (
            self._input._tensor[
                self._shuffle_tensor_indices[
                    [
                        self._input._indices_to_tensor_positions[index]
                        for index in range(batch_size * batch_number, max_index)
                    ]
                ]
            ].to(torch.float32)
            / 255
        )
        output_tensor: Tensor
        if isinstance(self._output, _SingleSizeImageList):
            output_tensor = (
                self._output._tensor[
                    self._shuffle_tensor_indices[
                        [
                            self._output._indices_to_tensor_positions[index]
                            for index in range(batch_size * batch_number, max_index)
                        ]
                    ]
                ].to(torch.float32)
                / 255
            )
        else:  # _output is instance of _TableAsTensor
            output_tensor = self._output._tensor[self._shuffle_tensor_indices[batch_size * batch_number : max_index]]
        return input_tensor, output_tensor

    def shuffle(self) -> ImageDataset[T]:
        """
        Return a new `ImageDataset` with shuffled data.

        The original dataset list is not modified.

        Returns
        -------
        image_dataset:
            the shuffled `ImageDataset`
        """
        import torch

        _init_default_device()

        im_dataset: ImageDataset[T] = copy.copy(self)
        im_dataset._shuffle_tensor_indices = torch.randperm(len(self))
        im_dataset._next_batch_index = 0
        return im_dataset


class _TableAsTensor:
    def __init__(self, table: Table) -> None:
        import polars as pl
        import torch

        _init_default_device()

        self._column_names = table.column_names
        if table.number_of_rows == 0:
            self._tensor = torch.empty((0, table.number_of_columns), dtype=torch.float32).to(_get_device())
        else:
            self._tensor = table._data_frame.to_torch(dtype=pl.Float32).to(_get_device())

        if not torch.all(self._tensor.sum(dim=1) == torch.ones(self._tensor.size(dim=0))):
            raise ValueError(
                "The given table is not correctly one hot encoded as it contains rows that have a sum not equal to 1.",
            )

    def __eq__(self, other: object) -> bool:
        import torch

        _init_default_device()

        if not isinstance(other, _TableAsTensor):
            return NotImplemented
        return (self is other) or (
            self._column_names == other._column_names and torch.all(torch.eq(self._tensor, other._tensor)).item()
        )

    def __hash__(self) -> int:
        return _structural_hash(self._tensor.size(), self._column_names)

    def __sizeof__(self) -> int:
        return (
            sys.getsizeof(self._tensor)
            + self._tensor.element_size() * self._tensor.nelement()
            + sys.getsizeof(self._column_names)
        )

    @staticmethod
    def _from_tensor(tensor: Tensor, column_names: list[str]) -> _TableAsTensor:
        if tensor.dim() != 2:
            raise ValueError(f"Tensor has an invalid amount of dimensions. Needed 2 dimensions but got {tensor.dim()}.")
        if tensor.size(dim=1) != len(column_names):
            raise ValueError(
                f"Tensor and column_names have different amounts of classes ({tensor.size(dim=1)}!={len(column_names)}).",
            )
        table_as_tensor = _TableAsTensor.__new__(_TableAsTensor)
        table_as_tensor._tensor = tensor
        table_as_tensor._column_names = column_names
        return table_as_tensor

    def _to_table(self) -> Table:
        return Table(dict(zip(self._column_names, self._tensor.T.tolist(), strict=False)))


class _ColumnAsTensor:
    def __init__(self, column: Column) -> None:
        import polars as pl
        import torch

        _init_default_device()

        self._column_name = column.name
        column_as_table = Table.from_columns([column])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=rf"The columns \['{self._column_name}'\] contain numerical data. The OneHotEncoder is designed to encode non-numerical values into numerical values",
                category=UserWarning,
            )
            # TODO: should not one-hot-encode the target. label encoding without order is sufficient. should also not
            #  be done automatically?
            self._one_hot_encoder = OneHotEncoder().fit(column_as_table, [self._column_name])
        self._tensor = torch.Tensor(
            self._one_hot_encoder.transform(column_as_table)._data_frame.to_torch(dtype=pl.Float32),
        ).to(_get_device())

    def __eq__(self, other: object) -> bool:
        import torch

        _init_default_device()

        if not isinstance(other, _ColumnAsTensor):
            return NotImplemented
        return (self is other) or (
            self._column_name == other._column_name
            and self._one_hot_encoder == other._one_hot_encoder
            and torch.all(torch.eq(self._tensor, other._tensor)).item()
        )

    def __hash__(self) -> int:
        return _structural_hash(self._tensor.size(), self._column_name, self._one_hot_encoder)

    def __sizeof__(self) -> int:
        return (
            sys.getsizeof(self._tensor)
            + self._tensor.element_size() * self._tensor.nelement()
            + sys.getsizeof(self._column_name)
            + sys.getsizeof(self._one_hot_encoder)
        )

    @staticmethod
    def _from_tensor(tensor: Tensor, column_name: str, one_hot_encoder: OneHotEncoder) -> _ColumnAsTensor:
        if tensor.dim() != 2:
            raise ValueError(f"Tensor has an invalid amount of dimensions. Needed 2 dimensions but got {tensor.dim()}.")
        if not one_hot_encoder.is_fitted:
            raise TransformerNotFittedError
        if tensor.size(dim=1) != len(one_hot_encoder._get_names_of_added_columns()):
            raise ValueError(
                f"Tensor and one_hot_encoder have different amounts of classes ({tensor.size(dim=1)}!={len(one_hot_encoder._get_names_of_added_columns())}).",
            )
        table_as_tensor = _ColumnAsTensor.__new__(_ColumnAsTensor)
        table_as_tensor._tensor = tensor
        table_as_tensor._column_name = column_name
        table_as_tensor._one_hot_encoder = one_hot_encoder
        return table_as_tensor

    def _to_column(self) -> Column:
        table = Table(
            dict(zip(self._one_hot_encoder._get_names_of_added_columns(), self._tensor.T.tolist(), strict=False)),
        )
        return self._one_hot_encoder.inverse_transform(table).get_column(self._column_name)
