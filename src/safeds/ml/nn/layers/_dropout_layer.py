from __future__ import annotations

from typing import TYPE_CHECKING, Any

from safeds._utils import _structural_hash
from safeds._validation import _check_bounds, _ClosedBound
from safeds.ml.nn.typing import ModelImageSize

from ._layer import Layer

if TYPE_CHECKING:
    from torch import nn


class DropoutLayer(Layer):
    """
    A dropout layer.

    Parameters
    ----------
    probability:
        The probability that an input element gets zeroed out.

    Raises
    ------
    OutOfBoundsError
        If probability is not in the range [0, 1].
    """

    def __init__(self, probability: float):
        _check_bounds("probability", probability, lower_bound=_ClosedBound(0), upper_bound=_ClosedBound(1))
        self._probability = probability
        self._input_size: int | ModelImageSize | None = None

    @property
    def probability(self) -> float:
        """The probability that an input element gets zeroed out."""
        return self._probability

    def _get_internal_layer(self, **_kwargs: Any) -> nn.Module:
        from ._internal_layers import _InternalDropoutLayer  # slow import on global level

        if self._input_size is None:
            raise ValueError(
                "The input_size is not yet set. The internal layer can only be created when the input_size is set.",
            )
        return _InternalDropoutLayer(self.probability)

    @property
    def input_size(self) -> int | ModelImageSize:
        """
        Get the input_size of this layer.

        Returns
        -------
        result:
            The amount of values being passed into this layer.

        Raises
        ------
        ValueError
            If the input_size is not yet set
        """
        if self._input_size is None:
            raise ValueError("The input_size is not yet set.")
        return self._input_size

    @property
    def output_size(self) -> int | ModelImageSize:
        """
        Get the output_size of this layer.

        Returns
        -------
        result:
            The amount of values being passed out of this layer.

        Raises
        ------
        ValueError
            If the input_size is not yet set
        """
        if self._input_size is None:
            raise ValueError("The input_size is not yet set.")
        return self._input_size

    def _set_input_size(self, input_size: int | ModelImageSize) -> None:
        self._input_size = input_size

    def _contains_choices(self) -> bool:
        return False

    def _get_layers_for_all_choices(self) -> list[DropoutLayer]:
        raise NotImplementedError  # pragma: no cover

    def __hash__(self) -> int:
        return _structural_hash(self._input_size, self._probability)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DropoutLayer):
            return NotImplemented
        if self is other:
            return True
        return self._input_size == other._input_size and self._probability == other._probability

    def __sizeof__(self) -> int:
        if self._input_size is None:
            raise ValueError("The input_size is not yet set.")
        if isinstance(self._input_size, int):
            return int(self._input_size)
        elif isinstance(self._input_size, ModelImageSize):
            return self._input_size.__sizeof__()
