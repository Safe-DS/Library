from pathlib import Path
from typing import TYPE_CHECKING

from torch import Tensor

from safeds.data.image.containers import Image
from safeds.data.image.containers._image_set import ImageSet

if TYPE_CHECKING:
    from safeds.data.image.containers import _FixedSizedImageSet


class _VariousSizedImageSet(ImageSet):

    def __init__(self):
        self._image_set_dict: dict[tuple[int, int], ImageSet] = {}

    @staticmethod
    def _create_image_set(images: list[Tensor], indices: list[int]) -> ImageSet:
        from safeds.data.image.containers import _FixedSizedImageSet

        max_channel = 0

        image_tensor_dict = {}
        image_index_dict = {}

        image_set = _VariousSizedImageSet()
        for index in indices:
            image = images.pop(0)
            size = (image.size(dim=2), image.size(dim=1))
            if size not in image_tensor_dict:
                image_tensor_dict[size] = [image]
                image_index_dict[size] = [index]
                max_channel = max(max_channel, image.size(dim=-3))
            else:
                image_tensor_dict[size].append(image)
                image_index_dict[size].append(index)
                max_channel = max(max_channel, image.size(dim=-3))

        for size in image_tensor_dict.keys():
            image_set._image_set_dict[size] = _FixedSizedImageSet._create_image_set(image_tensor_dict[size], image_index_dict[size])

        if max_channel > 1:
            image_set = image_set.change_channel(max_channel)

        return image_set

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageSet):
            return NotImplemented
        if not isinstance(other, _VariousSizedImageSet) or set(other._image_set_dict) != set(self._image_set_dict):
            return False
        for image_set_key, image_set_value in self._image_set_dict.items():
            if image_set_value != other._image_set_dict[image_set_key]:
                return False
        return True

    @property
    def number_of_images(self) -> int:
        length = 0
        for image_set in self._image_set_dict.values():
            length += len(image_set)
        return length

    @property
    def widths(self) -> list[int]:
        widths = []
        for image_set in self._image_set_dict.values():
            widths += image_set.widths
        return widths

    @property
    def heights(self) -> list[int]:
        heights = []
        for image_set in self._image_set_dict.values():
            heights += image_set.heights
        return heights

    @property
    def channel(self) -> int:
        for image_set in self._image_set_dict.values():
            return image_set.channel

    @property
    def number_of_sizes(self) -> int:
        return len(self._image_set_dict)

    @property
    def indices(self) -> list[int]:
        indices = []
        for image_set in self._image_set_dict.values():
            indices += image_set.indices
        return indices

    def get_image(self, index: int) -> Image:
        for image_set in self._image_set_dict.values():
            if index in image_set.indices:
                return image_set.get_image(index)
        raise KeyError(f'No image with index {index}')

    def index(self, image: Image) -> list[int]:
        indices = []
        for image_set in self._image_set_dict.values():
            if image in image_set:
                indices += image_set.index(image)
        return indices

    def has_image(self, image: Image) -> bool:
        return (image.width, image.height) in self._image_set_dict and self._image_set_dict[(image.width, image.height)].has_image(image)

    def to_jpeg_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    def to_png_files(self, path: str | Path | list[str] | list[Path]) -> None:
        pass

    def to_images(self, indices: list[int] | None = None) -> list[Image]:
        if indices is None:
            indices = self.indices
        image_list = []
        for image_set in self._image_set_dict.values():
            image_list += image_set.to_images(image_set.indices)
        return [image_list[indices.index(i)] for i in sorted(indices)]

    def change_channel(self, channel: int) -> ImageSet:
        image_set = _VariousSizedImageSet()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.change_channel(channel)
        return image_set

    def _add_image_tensor(self, image_tensor: Tensor, index: int) -> ImageSet:
        from safeds.data.image.containers import _FixedSizedImageSet

        image_set = _VariousSizedImageSet()
        image_set._image_set_dict = self._image_set_dict
        size = (image_tensor.size(dim=2), image_tensor.size(dim=1))

        current_max_channel = max_channel = self.channel

        if size in self._image_set_dict:
            image_set._image_set_dict[size] = image_set._image_set_dict[size]._add_image_tensor(image_tensor, index)
            max_channel = max(max_channel, image_set._image_set_dict[size].channel)
        else:
            image_set._image_set_dict[size] = _FixedSizedImageSet._create_image_set([image_tensor], [index])
            max_channel = max(max_channel, image_set._image_set_dict[size].channel)

        if max_channel > current_max_channel:
            image_set = image_set.change_channel(max_channel)

        return image_set

    def remove_image_by_index(self, index: int | list[int]) -> ImageSet:
        image_set = _VariousSizedImageSet()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set_new = image_set_original.remove_image_by_index(index)
            if len(image_set_new) > 0:
                image_set._image_set_dict[image_set_key] = image_set_new
        return image_set

    def remove_images_with_size(self, width: int, height: int) -> ImageSet:
        image_set = _VariousSizedImageSet()
        for image_set_key, image_set_original in self._image_set_dict.items():
            if (width, height) != image_set_key:
                image_set._image_set_dict[image_set_key] = image_set_original
        return image_set

    def shuffle_images(self) -> ImageSet:
        pass

    def resize(self, new_width: int, new_height: int) -> ImageSet:
        pass

    def convert_to_grayscale(self) -> ImageSet:
        pass

    def crop(self, x: int, y: int, width: int, height: int) -> ImageSet:
        pass

    def flip_vertically(self) -> ImageSet:
        pass

    def flip_horizontally(self) -> ImageSet:
        pass

    def adjust_brightness(self, factor: float) -> ImageSet:
        pass

    def add_noise(self, standard_deviation: float) -> ImageSet:
        pass

    def adjust_contrast(self, factor: float) -> ImageSet:
        pass

    def adjust_color_balance(self, factor: float) -> ImageSet:
        pass

    def blur(self, radius: int) -> ImageSet:
        pass

    def sharpen(self, factor: float) -> ImageSet:
        pass

    def invert_colors(self) -> ImageSet:
        pass

    def rotate_right(self) -> ImageSet:
        pass

    def rotate_left(self) -> ImageSet:
        pass

    def find_edges(self) -> ImageSet:
        pass
