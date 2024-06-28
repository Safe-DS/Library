from __future__ import annotations

import io
import math
import os
import random
from abc import ABCMeta, abstractmethod
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING, Literal, overload

from safeds._config import _init_default_device
from safeds._utils import _get_random_seed
from safeds._validation import _check_bounds, _ClosedBound
from safeds.data.image.containers._image import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
    from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
    from safeds.data.image.typing import ImageSize


class ImageList(metaclass=ABCMeta):
    """
    An ImageList is a list of different images. It can hold different sizes of Images. The channel of all images is the same.

    To create an `ImageList` call one of the following static methods:

    | Method                                                                        | Description                                              |
    | ----------------------------------------------------------------------------- | -------------------------------------------------------- |
    | [from_images][safeds.data.image.containers._image_list.ImageList.from_images] | Create an ImageList from a list of Images.               |
    | [from_files][safeds.data.image.containers._image_list.ImageList.from_files]   | Create an ImageList from a directory or a list of files. |
    """

    @staticmethod
    @abstractmethod
    def _create_image_list(images: list[Tensor], indices: list[int]) -> ImageList:
        """
        Create an ImageList from a list of tensors.

        Parameters
        ----------
        images:
            the list of tensors
        indices:
            a list of indices for the tensors

        Returns
        -------
        image_list:
            The image list
        """

    @staticmethod
    def from_images(images: list[Image]) -> ImageList:
        """
        Create an ImageList from a list of images.

        Parameters
        ----------
        images:
            the list of images

        Returns
        -------
        image_list:
            the image list
        """
        from safeds.data.image.containers._empty_image_list import _EmptyImageList
        from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
        from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList

        if len(images) == 0:
            return _EmptyImageList()

        indices = list(range(len(images)))
        first_width = images[0].width
        first_height = images[0].height
        for im in images:
            if first_width != im.width or first_height != im.height:
                return _MultiSizeImageList._create_image_list([image._image_tensor for image in images], indices)
        return _SingleSizeImageList._create_image_list([image._image_tensor for image in images], indices)

    @staticmethod
    @overload
    def from_files(path: str | Path | Sequence[str | Path]) -> ImageList: ...

    @staticmethod
    @overload
    def from_files(path: str | Path | Sequence[str | Path], *, load_percentage: float) -> ImageList: ...

    @staticmethod
    @overload
    def from_files(path: str | Path | Sequence[str | Path], *, return_filenames: Literal[False]) -> ImageList: ...

    @staticmethod
    @overload
    def from_files(
        path: str | Path | Sequence[str | Path],
        *,
        return_filenames: Literal[False],
        load_percentage: float,
    ) -> ImageList: ...

    @staticmethod
    @overload
    def from_files(
        path: str | Path | Sequence[str | Path],
        *,
        return_filenames: Literal[True],
    ) -> tuple[ImageList, list[str]]: ...

    @staticmethod
    @overload
    def from_files(
        path: str | Path | Sequence[str | Path],
        *,
        return_filenames: Literal[True],
        load_percentage: float,
    ) -> tuple[ImageList, list[str]]: ...

    @staticmethod
    @overload
    def from_files(
        path: str | Path | Sequence[str | Path],
        *,
        return_filenames: bool,
    ) -> ImageList | tuple[ImageList, list[str]]: ...

    @staticmethod
    @overload
    def from_files(
        path: str | Path | Sequence[str | Path],
        *,
        return_filenames: bool,
        load_percentage: float,
    ) -> ImageList | tuple[ImageList, list[str]]: ...

    @staticmethod
    def from_files(
        path: str | Path | Sequence[str | Path],
        *,
        return_filenames: bool = False,
        load_percentage: float = 1.0,
    ) -> ImageList | tuple[ImageList, list[str]]:
        """
        Create an ImageList from a directory or a list of files.

        If you provide a path to a directory the images will be sorted alphabetically while inner directories will be sorted after image files.

        Parameters
        ----------
        path:
            the path to the directory or a list of files
        return_filenames:
            if True the output will be a tuple which contains a list of the filenames in order of the images
        load_percentage:
            the percentage of the given data being loaded. If below 1 the files will be shuffled before loading

        Returns
        -------
        image_list:
            the image list

        Raises
        ------
        FileNotFoundError
            If the directory or one of the files of the path cannot be found
        OutOfBoundsError
            If load_percentage is not between 0 and 1
        """
        from PIL.Image import open as pil_image_open

        _init_default_device()

        random.seed(_get_random_seed())

        from safeds.data.image.containers._empty_image_list import _EmptyImageList
        from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
        from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList

        _check_bounds("load_percentage", load_percentage, lower_bound=_ClosedBound(0), upper_bound=_ClosedBound(1))

        if isinstance(path, list) and len(path) == 0:
            return _EmptyImageList()

        file_names = []

        path_list: list[str | Path]
        if isinstance(path, Path | str):
            path_list = [Path(path)]
        else:
            path_list = list(path)
        while len(path_list) != 0:
            p = Path(path_list.pop(0))
            if p.is_dir():
                path_list += sorted([p / name for name in os.listdir(p)])
            elif p.is_file():
                file_names.append(str(p))
            else:
                raise FileNotFoundError(f"No such file or directory: '{path}'")

        if load_percentage < 1:
            random.shuffle(file_names)
            file_names = file_names[: max(round(len(file_names) * load_percentage), 1) if load_percentage > 0 else 0]

        num_of_files = len(file_names)

        if num_of_files == 0:
            return _EmptyImageList()

        image_sizes: dict[tuple[int, int], dict[int, list[str]]] = {}
        image_indices: dict[tuple[int, int], dict[int, list[int]]] = {}
        image_count: dict[tuple[int, int], int] = {}
        max_channel = -1

        for i, filename in enumerate(file_names):
            im = pil_image_open(filename)
            im_channel = len(im.getbands())
            im_size = (im.width, im.height)
            if im_channel > max_channel:
                max_channel = im_channel
            if im_size not in image_sizes:
                image_sizes[im_size] = {im_channel: [filename]}
                image_indices[im_size] = {im_channel: [i]}
                image_count[im_size] = 1
            elif im_channel not in image_sizes[im_size]:
                image_sizes[im_size][im_channel] = [filename]
                image_indices[im_size][im_channel] = [i]
                image_count[im_size] += 1
            else:
                image_sizes[im_size][im_channel].append(filename)
                image_indices[im_size][im_channel].append(i)
                image_count[im_size] += 1

        num_of_threads = min(math.ceil(num_of_files / 1000), 100)
        num_of_files_per_thread = math.ceil(num_of_files / num_of_threads)

        single_sized_image_lists = []
        thread_packages = []
        for size, image_files in image_sizes.items():
            im_list, packages = _SingleSizeImageList._create_image_list_from_files(
                image_files,
                image_count[size],
                max_channel,
                size[0],
                size[1],
                image_indices[size],
                num_of_files_per_thread,
            )
            single_sized_image_lists.append(im_list._as_single_size_image_list())
            thread_packages += packages
        thread_packages.sort(key=lambda x: len(x), reverse=True)

        threads: list[ImageList._FromImageThread] = []
        for thread_index in range(num_of_threads):
            current_thread_workload = 0
            current_thread_packages = []
            while current_thread_workload < num_of_files_per_thread and len(thread_packages) > 0:
                next_package = thread_packages.pop()
                current_thread_packages.append(next_package)
                current_thread_workload += len(next_package)
            if thread_index == num_of_threads - 1 and len(thread_packages) > 0:
                current_thread_packages += thread_packages  # pragma: no cover
            thread = ImageList._FromImageThread(current_thread_packages)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        if len(single_sized_image_lists) == 1:
            image_list: ImageList = single_sized_image_lists[0]
        else:
            image_list = _MultiSizeImageList._create_from_single_sized_image_lists(single_sized_image_lists)

        if return_filenames:
            return image_list, file_names
        else:
            return image_list

    class _FromFileThreadPackage:
        def __init__(
            self,
            im_files: list[str],
            im_channel: int,
            to_channel: int,
            im_width: int,
            im_height: int,
            tensor: Tensor,
            start_index: int,
        ) -> None:
            self._im_files = im_files
            self._im_channel = im_channel
            self._to_channel = to_channel
            self._im_width = im_width
            self._im_height = im_height
            self._tensor = tensor
            self._start_index = start_index

        def load_files(self) -> None:
            import torch
            from torchvision.io import read_image

            _init_default_device()

            num_of_files = len(self._im_files)
            tensor_channel = max(self._im_channel, min(self._to_channel, 3))
            for index, im in enumerate(self._im_files):
                self._tensor[index + self._start_index, 0:tensor_channel] = read_image(im)
            if self._to_channel == 4 and self._im_channel < 4:
                torch.full(
                    (num_of_files, 1, self._im_height, self._im_width),
                    255,
                    out=self._tensor[self._start_index : self._start_index + num_of_files, 3:4],
                )

        def __len__(self) -> int:
            return len(self._im_files)

    class _FromImageThread(Thread):
        def __init__(self, packages: list[ImageList._FromFileThreadPackage]) -> None:
            super().__init__()
            self._packages = packages

        def run(self) -> None:
            for pck in self._packages:
                pck.load_files()

    @abstractmethod
    def _clone(self) -> ImageList:
        """
        Clone your ImageList to a new instance.

        Returns
        -------
        image_list:
            the cloned image list
        """

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """
        Compare two image lists.

        Two image lists are only equal if they contain the same images at the same indices.

        Parameters
        ----------
        other:
            the image list to compare to

        Returns
        -------
        equals:
            Whether the two image lists are equal
        """

    @abstractmethod
    def __hash__(self) -> int:
        """
        Return a deterministic hash value for this image list.

        Returns
        -------
        hash:
            The hash value.
        """

    @abstractmethod
    def __sizeof__(self) -> int:
        """
        Return the complete size of this object.

        Returns
        -------
        size:
            Size of this object in bytes.
        """

    def __len__(self) -> int:
        """
        Return the number of images in this image list.

        Returns
        -------
        length:
            The number of images
        """
        return self.image_count

    def __contains__(self, item: object) -> bool:
        """
        Return whether the given item is in this image list.

        Parameters
        ----------
        item:
            the item to check

        Returns
        -------
        has_item:
            Whether the given item is in this image list
        """
        return isinstance(item, Image) and self.has_image(item)

    def _repr_png_(self) -> bytes:
        """
        Return a PNG representation of this image list as bytes.

        Returns
        -------
        png:
            the png representation of this image list
        """
        import torch
        from torchvision.utils import make_grid, save_image

        _init_default_device()

        from safeds.data.image.containers._empty_image_list import _EmptyImageList

        if isinstance(self, _EmptyImageList):
            raise TypeError("You cannot display an empty ImageList")

        max_width, max_height = max(self.widths), max(self.heights)
        tensors = []
        for image in self.to_images():
            im_tensor = torch.zeros([4, max_height, max_width])
            im_tensor[:, : image.height, : image.width] = image.change_channel(4)._image_tensor
            tensors.append(im_tensor)
        tensor_grid = make_grid(tensors, math.ceil(math.sqrt(len(tensors))))
        buffer = io.BytesIO()
        save_image(tensor_grid.to(torch.float32) / 255, buffer, format="png")
        buffer.seek(0)
        return buffer.read()

    # ------------------------------------------------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def image_count(self) -> int:
        """The number of images in this image list."""

    @property
    @abstractmethod
    def widths(self) -> list[int]:
        """A list of all widths in this image list."""

    @property
    @abstractmethod
    def heights(self) -> list[int]:
        """A list of all heights in this image list."""

    @property
    @abstractmethod
    def channel(self) -> int:
        """The channel of all images."""

    @property
    @abstractmethod
    def sizes(self) -> list[ImageSize]:
        """The sizes of all images."""

    @property
    @abstractmethod
    def size_count(self) -> int:
        """The number of different sizes of images in this image list."""

    # ------------------------------------------------------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def get_image(self, index: int) -> Image:
        """
        Return the image at the given index.

        Parameters
        ----------
        index:
            the index for the image to return

        Returns
        -------
        image:
            the image at the given index
        """

    @abstractmethod
    def index(self, image: Image) -> list[int]:
        """
        Return a list of indexes of the given image.

        If the image has multiple occurrences, all indices will be returned

        Parameters
        ----------
        image:
            the image to search for occurrences

        Returns
        -------
        indices:
            all occurrences of the image
        """

    @abstractmethod
    def has_image(self, image: Image) -> bool:
        """
        Return whether the given image is in this image list.

        Parameters
        ----------
        image:
            the image to check

        Returns
        -------
        has_image:
            Whether the given image is in this image list
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def to_jpeg_files(self, path: str | Path | list[str | Path]) -> None:
        """
        Save all images as jpeg files.

        Parameters
        ----------
        path:
            Either the path to a directory or a list of directories which has directories for either all different sizes or all different images. Any non-existant path will be created

        Raises
        ------
        IllegalFormatError
            If the channel of the images is not supported
        ValueError
            If the path is a list but has too few or too many entries
        """

    @abstractmethod
    def to_png_files(self, path: str | Path | list[str | Path]) -> None:
        """
        Save all images as png files.

        Parameters
        ----------
        path:
            Either the path to a directory or a list of directories which has directories for either all different sizes or all different images. Any non-existant path will be created

        Raises
        ------
        ValueError
            If the path is a list but has too few or too many entries
        """

    @abstractmethod
    def to_images(self, indices: list[int] | None = None) -> list[Image]:
        """
        Return a list of all images in this image list.

        Parameters
        ----------
        indices:
            a list of all indices to include in the output. If None, all indices will be included

        Returns
        -------
        images:
            the list of all images

        Raises
        ------
        IndexOutOfBoundsError
            If any index is out of bounds
        """

    def _as_multi_size_image_list(self) -> _MultiSizeImageList:
        """
        Typechecking method for MultiSizeImageList.

        Returns
        -------
        self:
            self as a MultiSizeImageList

        Raises
        ------
        ValueError
            if this image list is not a MultiSizeImageList
        """
        from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList

        if isinstance(self, _MultiSizeImageList):
            return self
        raise ValueError("The given image_list is not a MultiSizeImageList")

    def _as_single_size_image_list(self) -> _SingleSizeImageList:
        """
        Typechecking method for SingleSizeImageList.

        Returns
        -------
        self:
            self as a SingleSizeImageList

        Raises
        ------
        ValueError
            if this image list is not a SingleSizeImageList
        """
        from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList

        if isinstance(self, _SingleSizeImageList):
            return self
        raise ValueError("The given image_list is not a SingleSizeImageList")

    # ------------------------------------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def change_channel(self, channel: int) -> ImageList:
        """
        Return a new `ImageList` that has the given number of channels.

        The original image list is not modified.

        Parameters
        ----------
        channel:
            The new number of channels. 1 will result in grayscale images

        Returns
        -------
        image_list:
            the image list with the given number of channels

        Raises
        ------
        ValueError:
            if the given channel is not a valid channel option
        """

    @abstractmethod
    def _add_image_tensor(self, image_tensor: Tensor, index: int) -> ImageList:
        """
        Return a new `ImageList` with the given tensor added as an image.

        The original image list is not modified.

        Parameters
        ----------
        image_tensor:
            The new tensor to be added to the image list

        Returns
        -------
        image_list:
            the image list with the new tensor added
        """

    def add_image(self, image: Image) -> ImageList:
        """
        Return a new `ImageList` with the given image added to the image list.

        The original image list is not modified.

        Parameters
        ----------
        image:
            The image to be added to the image list

        Returns
        -------
        image_list:
            the image list with the new image added
        """
        return self._add_image_tensor(image._image_tensor, self.image_count)

    @abstractmethod
    def add_images(self, images: list[Image] | ImageList) -> ImageList:
        """
        Return a new `ImageList` with the given images added to the image list.

        The original image list is not modified.

        Parameters
        ----------
        images:
            The images to be added to the image list

        Returns
        -------
        image_list:
            the image list with the new images added
        """

    def remove_image(self, image: Image) -> ImageList:
        """
        Return a new `ImageList` with the given image removed from the image list.

        If the image has multiple occurrences, all occurrences will be removed.

        The original image list is not modified.

        Parameters
        ----------
        image:
            The image to be removed from the image list

        Returns
        -------
        image_list:
            the image list with the given image removed
        """
        return self._remove_image_by_index_ignore_invalid(self.index(image))

    def remove_images(self, images: list[Image]) -> ImageList:
        """
        Return a new `ImageList` with the given images removed from the image list.

        If one image has multiple occurrences, all occurrences will be removed.

        The original image list is not modified.

        Parameters
        ----------
        images:
            The images to be removed from the image list

        Returns
        -------
        image_list:
            the image list with the given images removed
        """
        indices_to_remove = []
        for image in images:
            indices_to_remove += self.index(image)
        return self._remove_image_by_index_ignore_invalid(list(set(indices_to_remove)))

    @abstractmethod
    def remove_image_by_index(self, index: int | list[int]) -> ImageList:
        """
        Return a new `ImageList` with the given indices removed from the image list.

        The original image list is not modified.

        Parameters
        ----------
        index:
            The index of the image to be removed from the image list

        Returns
        -------
        image_list:
            the image list with the without the removed image

        Raises
        ------
        IndexOutOfBoundsError
            If one of the indices is out of bounds
        """

    @abstractmethod
    def _remove_image_by_index_ignore_invalid(self, index: int | list[int]) -> ImageList:
        """
        Return a new `ImageList` with the given indices removed from the image list.

        Invalid indices will be ignored.

        The original image list is not modified.

        Parameters
        ----------
        index:
            The index of the image to be removed from the image list

        Returns
        -------
        image_list:
            the image list with the without the removed image
        """

    @abstractmethod
    def remove_images_with_size(self, width: int, height: int) -> ImageList:
        """
        Return a new `ImageList` with the all images of the given size removed.

        The original image list is not modified.

        Parameters
        ----------
        width:
            The width of the images to be removed from the image list
        height:
            The height of the images to be removed from the image list

        Returns
        -------
        image_list:
            the image list with the given images removed

        Raises
        ------
        OutOfBoundsError
            If width or height are below 1
        """

    @abstractmethod
    def remove_duplicate_images(self) -> ImageList:
        """
        Return a new `ImageList` with all duplicate images removed.

        One occurrence of each image will stay in the image list.

        The original image list is not modified.

        Returns
        -------
        image_list:
            the image list with only unique images
        """

    @abstractmethod
    def shuffle_images(self) -> ImageList:
        """
        Return a new `ImageList` with all images shuffled.

        The original image list is not modified.

        Returns
        -------
        image_list:
            the image list with shuffled images
        """

    @abstractmethod
    def resize(self, new_width: int, new_height: int) -> ImageList:
        """
        Return a new `ImageList` with all images resized to a given size.

        The original image list is not modified.

        Parameters
        ----------
        new_width:
            the new width of the images
        new_height:
            the new height of the images

        Returns
        -------
        image_list:
            The image list with all images resized to the given width and height.

        Raises
        ------
        OutOfBoundsError
            If new_width or new_height are below 1
        """

    @abstractmethod
    def convert_to_grayscale(self) -> ImageList:
        """
        Return a new `ImageList` with all images converted to grayscale.

        The new image list will have the same amount of channels as the original image list.
        If you want to change the amount of channels used, please use the method [change_channel][safeds.data.image.containers._image_list.ImageList.change_channel].

        The original image list is not modified.

        Returns
        -------
        image_list:
            The image list with all images converted to grayscale.
        """

    @abstractmethod
    def crop(self, x: int, y: int, width: int, height: int) -> ImageList:
        """
        Return a new `ImageList` with all images cropped to a given bounding rectangle.

        The original image list is not modified.

        Parameters
        ----------
        x:
            the x coordinate of the top-left corner of the bounding rectangle
        y:
            the y coordinate of the top-left corner of the bounding rectangle
        width:
            the width of the bounding rectangle
        height:
            the height of the bounding rectangle

        Returns
        -------
        image_list:
            The image list with all images cropped

        Raises
        ------
        OutOfBoundsError
            If x or y are below 0 or if width or height are below 1
        """

    @abstractmethod
    def flip_vertically(self) -> ImageList:
        """
        Return a new `ImageList` with all images flipped vertically (horizontal axis, flips up-down and vice versa).

        The original image list is not modified.

        Returns
        -------
        image_list:
            The image list with all images flipped vertically
        """

    @abstractmethod
    def flip_horizontally(self) -> ImageList:
        """
        Return a new `ImageList` with all images flipped horizontally (vertical axis, flips left-right and vice versa).

        The original image list is not modified.

        Returns
        -------
        image_list:
            The image list with all images flipped horizontally
        """

    @abstractmethod
    def adjust_brightness(self, factor: float) -> ImageList:
        """
        Return a new `ImageList` where all images have the adjusted brightness.

        The original image list is not modified.

        Parameters
        ----------
        factor:
            The brightness factor.
            1.0 will not change the brightness.
            Below 1.0 will result in a darker images.
            Above 1.0 will resolut in a brighter images.
            Has to be bigger than or equal to 0 (black).

        Returns
        -------
        image_list:
            The image list with adjusted brightness

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """

    @abstractmethod
    def add_noise(self, standard_deviation: float) -> ImageList:
        """
        Return a new `ImageList` with noise added to all images.

        The original image list is not modified.

        Parameters
        ----------
        standard_deviation:
            The standard deviation of the normal distribution. Has to be bigger than or equal to 0.

        Returns
        -------
        image_list:
            The image list with added noise

        Raises
        ------
        OutOfBoundsError
            If standard_deviation is smaller than 0.
        """

    @abstractmethod
    def adjust_contrast(self, factor: float) -> ImageList:
        """
        Return a new `ImageList` where all images have the adjusted contrast.

        The original image list is not modified.

        Parameters
        ----------
        factor:
            If factor > 1, increase contrast of images.
            If factor = 1, no changes will be made.
            If factor < 1, make images greyer.
            Has to be bigger than or equal to 0 (gray).

        Returns
        -------
        image_list:
            The image list with adjusted contrast

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """

    @abstractmethod
    def adjust_color_balance(self, factor: float) -> ImageList:
        """
        Return a new `ImageList` where all images have the adjusted color balance.

        The original image list is not modified.

        Parameters
        ----------
        factor:
            Has to be bigger than or equal to 0.
            If 0 <= factor < 1, make images greyer.
            If factor = 1, no changes will be made.
            If factor > 1, increase color balance of images.

        Returns
        -------
        image_list:
            The image list with adjusted color balance

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """

    @abstractmethod
    def blur(self, radius: int) -> ImageList:
        """
        Return a new `ImageList` where all images have been blurred.

        The original image list is not modified.

        Parameters
        ----------
        radius:
             Radius is directly proportional to the blur value. The radius is equal to the amount of pixels united in
             each direction. A radius of 1 will result in a united box of 9 pixels.

        Returns
        -------
        image_list:
            The image list with blurred images

        Raises
        ------
        OutOfBoundsError
            If radius is smaller than 0 or equal or greater than the smallest size of one of the images.
        """

    @abstractmethod
    def sharpen(self, factor: float) -> ImageList:
        """
        Return a new `ImageList` where all images have been sharpened.

        The original image list is not modified.

        Parameters
        ----------
        factor:
            If factor > 1, increase the sharpness of the images.
            If factor = 1, no changes will be made.
            If factor < 1, blur the images.
            Has to be bigger than or equal to 0 (blurred).

        Returns
        -------
        image_list:
            The image list with sharpened images

        Raises
        ------
        OutOfBoundsError
            If factor is smaller than 0.
        """

    @abstractmethod
    def invert_colors(self) -> ImageList:
        """
        Return a new `ImageList` where all images have their colors inverted.

        The original image list is not modified.

        Returns
        -------
        image_list:
            The image list with inverted colors
        """

    @abstractmethod
    def rotate_right(self) -> ImageList:
        """
        Return a new `ImageList` where all images have been rotated 90 degrees clockwise.

        The original image list is not modified.

        Returns
        -------
        image_list:
            The image list with all images rotated
        """

    @abstractmethod
    def rotate_left(self) -> ImageList:
        """
        Return a new `ImageList` where all images have been rotated 90 degrees counter-clockwise.

        The original image list is not modified.

        Returns
        -------
        image_list:
            The image list with all images rotated
        """

    @abstractmethod
    def find_edges(self) -> ImageList:
        """
        Return a new `ImageList` with grayscale versions of the images with the edges highlighted.

        The original image list is not modified.

        Returns
        -------
        image_list:
            The image list with highlighted edges
        """
