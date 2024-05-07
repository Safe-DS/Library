import math
import random
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch.types import Device

from safeds.data.image.containers import Image, ImageList
from safeds.data.image.containers._empty_image_list import _EmptyImageList
from safeds.data.image.containers._multi_size_image_list import _MultiSizeImageList
from safeds.data.image.containers._single_size_image_list import _SingleSizeImageList
from safeds.data.tabular.containers import Table
from safeds.exceptions import DuplicateIndexError, IllegalFormatError, IndexOutOfBoundsError, OutOfBoundsError
from syrupy import SnapshotAssertion
from torch import Tensor

from tests.helpers import (
    grayscale_jpg_path,
    grayscale_png_path,
    images_all,
    images_all_channel,
    images_all_channel_ids,
    images_all_ids,
    os_mac,
    plane_jpg_path,
    plane_png_path,
    resolve_resource_path,
    skip_if_os,
    test_images_folder,
    white_square_jpg_path, get_devices, get_devices_ids, configure_test_with_device,
)


@pytest.mark.parametrize("resource_path3", images_all(), ids=images_all_ids())
@pytest.mark.parametrize("resource_path2", images_all(), ids=images_all_ids())
@pytest.mark.parametrize("resource_path1", images_all(), ids=images_all_ids())
@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestAllImageCombinations:

    def test_from_files(self, resource_path1: str, resource_path2: str, resource_path3: str, device: Device) -> None:
        # Setup
        configure_test_with_device(device)

        image_list = ImageList.from_files(
            [
                resolve_resource_path(resource_path1),
                resolve_resource_path(resource_path2),
                resolve_resource_path(resource_path3),
            ],
        )
        image1 = Image.from_file(resolve_resource_path(resource_path1))
        image2 = Image.from_file(resolve_resource_path(resource_path2))
        image3 = Image.from_file(resolve_resource_path(resource_path3))
        expected_channel = max(image1.channel, image2.channel, image3.channel)
        images_not_included = []
        for image_path in images_all():
            if image_path[:-4] not in (resource_path1[:-4], resource_path2[:-4], resource_path3[:-4]):
                images_not_included.append(
                    Image.from_file(resolve_resource_path(image_path)).change_channel(expected_channel),
                )

        image1_with_expected_channel = image1.change_channel(expected_channel)
        image2_with_expected_channel = image2.change_channel(expected_channel)
        image3_with_expected_channel = image3.change_channel(expected_channel)

        # Test clone
        image_list_clone = image_list._clone()
        assert image_list_clone is not image_list
        assert image_list_clone == image_list

        # Check creation of EmptyImageList
        assert image_list == _EmptyImageList().add_images([image1, image2, image3])
        assert image_list == _EmptyImageList().add_images(image_list)

        # Check if factory method selected the right ImageList
        if image1.width == image2.width == image3.width and image1.height == image2.height == image3.height:
            assert isinstance(image_list, _SingleSizeImageList)
            assert isinstance(image_list._as_single_size_image_list(), _SingleSizeImageList)
            with pytest.raises(ValueError, match=r"The given image_list is not a MultiSizeImageList"):
                image_list._as_multi_size_image_list()
        else:
            assert isinstance(image_list, _MultiSizeImageList)
            assert isinstance(image_list._as_multi_size_image_list(), _MultiSizeImageList)
            with pytest.raises(ValueError, match=r"The given image_list is not a SingleSizeImageList"):
                image_list._as_single_size_image_list()

            # Check if all children are SingleSizeImageLists and have the right channel count
            for fixed_image_list in image_list._image_list_dict.values():
                assert isinstance(fixed_image_list, _SingleSizeImageList)
                assert fixed_image_list.channel == expected_channel

        # Check if all images have the right index
        assert 0 in image_list.index(image1_with_expected_channel)
        assert 1 in image_list.index(image2_with_expected_channel)
        assert 2 in image_list.index(image3_with_expected_channel)
        for image in images_not_included:
            assert not image_list.index(image)
        with pytest.raises(IndexOutOfBoundsError, match=r"There is no element at index '3'."):
            image_list.get_image(3)
        assert image_list.get_image(0) == image1_with_expected_channel
        assert image_list.get_image(1) == image2_with_expected_channel
        assert image_list.get_image(2) == image3_with_expected_channel

        # Test eq
        image_list_equal = ImageList.from_files(
            [
                resolve_resource_path(resource_path1),
                resolve_resource_path(resource_path2),
                resolve_resource_path(resource_path3),
            ],
        )
        image_list_unequal_1 = ImageList.from_images(images_not_included)
        image_list_unequal_2 = image_list.remove_image_by_index(0)
        image_list_unequal_3 = image_list._remove_image_by_index_ignore_invalid(0)
        assert image_list == image_list_equal
        assert image_list != image_list_unequal_1
        assert image_list != image_list_unequal_2
        assert image_list != image_list_unequal_3
        assert image_list.__eq__(Table()) is NotImplemented

        # Test hash
        assert hash(image_list) == hash(image_list_clone)
        assert hash(image_list) == hash(image_list_equal)
        assert hash(image_list) != hash(image_list_unequal_1)
        assert hash(image_list) != hash(image_list_unequal_2)

        # Test size
        assert sys.getsizeof(image_list) >= image1.__sizeof__() + image2.__sizeof__() + image3.__sizeof__()

        # Test from_images
        image_list_from_images = ImageList.from_images([image1, image2, image3])
        assert image_list_from_images is not image_list
        assert image_list_from_images == image_list

        # Test len
        assert len(image_list) == 3

        # Test contains
        assert image1_with_expected_channel in image_list
        assert image2_with_expected_channel in image_list
        assert image3_with_expected_channel in image_list
        for image in images_not_included:
            assert image not in image_list

        # Test number_of_images
        assert image_list.number_of_images == 3

        # Test widths
        assert image_list.widths == [image1.width, image2.width, image3.width]

        # Test heights
        assert image_list.heights == [image1.height, image2.height, image3.height]

        # Test channel
        assert image_list.channel == expected_channel

        # Test sizes
        assert image_list.sizes == [
            image1_with_expected_channel.size,
            image2_with_expected_channel.size,
            image3_with_expected_channel.size,
        ]

        # Test number_of_sizes
        assert image_list.number_of_sizes == len({(image.width, image.height) for image in [image1, image2, image3]})

        # Test has_image
        assert image_list.has_image(image1_with_expected_channel)
        assert image_list.has_image(image2_with_expected_channel)
        assert image_list.has_image(image3_with_expected_channel)
        for image in images_not_included:
            assert not image_list.has_image(image)

        # Test to_images
        assert image_list.to_images() == [
            image1_with_expected_channel,
            image2_with_expected_channel,
            image3_with_expected_channel,
        ]

        # Test change_channel
        assert image_list.change_channel(1).channel == 1
        assert image_list.change_channel(3).channel == 3
        assert image_list.change_channel(4).channel == 4

        # Test add image
        assert image_list == ImageList.from_images([image1]).add_image(image2).add_image(image3)

        # Test add images
        assert image_list == ImageList.from_images([image1]).add_images([image2, image3])
        assert image_list == ImageList.from_images([image1, image2]).add_images([image3])
        assert image_list == ImageList.from_images([image1]).add_images(ImageList.from_images([image2, image3]))
        assert image_list == ImageList.from_images([image1, image2]).add_images(ImageList.from_images([image3]))
        assert ImageList.from_images([image1, image2, image3, *images_not_included]) == image_list.add_images(
            images_not_included,
        )
        assert image_list == image_list.add_images([])
        assert image_list == image_list.add_images(_EmptyImageList())

        # Test remove image
        image_list_wo_im_1 = image_list.remove_image(image1_with_expected_channel)
        image_list_wo_im_2 = image_list.remove_image(image2_with_expected_channel)
        image_list_wo_im_3 = image_list.remove_image(image3_with_expected_channel)

        assert image_list_wo_im_1 == ImageList.from_images(
            [
                im
                for im in [image2_with_expected_channel, image3_with_expected_channel]
                if im != image1_with_expected_channel
            ],
        )
        assert image_list_wo_im_2 == ImageList.from_images(
            [
                im
                for im in [image1_with_expected_channel, image3_with_expected_channel]
                if im != image2_with_expected_channel
            ],
        )
        assert image_list_wo_im_3 == ImageList.from_images(
            [
                im
                for im in [image1_with_expected_channel, image2_with_expected_channel]
                if im != image3_with_expected_channel
            ],
        )
        assert image_list_wo_im_1.remove_image(image2_with_expected_channel) == ImageList.from_images(
            (
                [image3_with_expected_channel]
                if image3_with_expected_channel not in [image1_with_expected_channel, image2_with_expected_channel]
                else []
            ),
        )
        assert image_list_wo_im_2.remove_image(image1_with_expected_channel) == ImageList.from_images(
            (
                [image3_with_expected_channel]
                if image3_with_expected_channel not in [image1_with_expected_channel, image2_with_expected_channel]
                else []
            ),
        )
        assert image_list_wo_im_1.remove_image(image3_with_expected_channel) == ImageList.from_images(
            (
                [image2_with_expected_channel]
                if image2_with_expected_channel not in [image1_with_expected_channel, image3_with_expected_channel]
                else []
            ),
        )
        assert image_list_wo_im_3.remove_image(image1_with_expected_channel) == ImageList.from_images(
            (
                [image2_with_expected_channel]
                if image2_with_expected_channel not in [image1_with_expected_channel, image3_with_expected_channel]
                else []
            ),
        )
        assert image_list_wo_im_2.remove_image(image3_with_expected_channel) == ImageList.from_images(
            (
                [image1_with_expected_channel]
                if image1_with_expected_channel not in [image2_with_expected_channel, image3_with_expected_channel]
                else []
            ),
        )
        assert image_list_wo_im_3.remove_image(image2_with_expected_channel) == ImageList.from_images(
            (
                [image1_with_expected_channel]
                if image1_with_expected_channel not in [image2_with_expected_channel, image3_with_expected_channel]
                else []
            ),
        )
        assert (
            image_list_wo_im_1.remove_image(image2_with_expected_channel).remove_image(image3_with_expected_channel)
            == _EmptyImageList()
        )
        assert (
            image_list_wo_im_1.remove_image(image3_with_expected_channel).remove_image(image2_with_expected_channel)
            == _EmptyImageList()
        )
        assert (
            image_list_wo_im_2.remove_image(image1_with_expected_channel).remove_image(image3_with_expected_channel)
            == _EmptyImageList()
        )
        assert (
            image_list_wo_im_2.remove_image(image3_with_expected_channel).remove_image(image1_with_expected_channel)
            == _EmptyImageList()
        )
        assert (
            image_list_wo_im_3.remove_image(image1_with_expected_channel).remove_image(image2_with_expected_channel)
            == _EmptyImageList()
        )
        assert (
            image_list_wo_im_3.remove_image(image2_with_expected_channel).remove_image(image1_with_expected_channel)
            == _EmptyImageList()
        )

        # # Test remove images
        assert image_list.remove_images([image1_with_expected_channel]) == ImageList.from_images(
            [
                im
                for im in [image2_with_expected_channel, image3_with_expected_channel]
                if im != image1_with_expected_channel
            ],
        )
        assert image_list.remove_images([image2_with_expected_channel]) == ImageList.from_images(
            [
                im
                for im in [image1_with_expected_channel, image3_with_expected_channel]
                if im != image2_with_expected_channel
            ],
        )
        assert image_list.remove_images([image3_with_expected_channel]) == ImageList.from_images(
            [
                im
                for im in [image1_with_expected_channel, image2_with_expected_channel]
                if im != image3_with_expected_channel
            ],
        )
        assert image_list.remove_images(
            [image1_with_expected_channel, image2_with_expected_channel],
        ) == ImageList.from_images(
            (
                [image3_with_expected_channel]
                if image3_with_expected_channel not in [image1_with_expected_channel, image2_with_expected_channel]
                else []
            ),
        )
        assert image_list.remove_images(
            [image1_with_expected_channel, image3_with_expected_channel],
        ) == ImageList.from_images(
            (
                [image2_with_expected_channel]
                if image2_with_expected_channel not in [image1_with_expected_channel, image3_with_expected_channel]
                else []
            ),
        )
        assert image_list.remove_images(
            [image2_with_expected_channel, image3_with_expected_channel],
        ) == ImageList.from_images(
            (
                [image1_with_expected_channel]
                if image1_with_expected_channel not in [image2_with_expected_channel, image3_with_expected_channel]
                else []
            ),
        )
        assert (
            image_list.remove_images(
                [image1_with_expected_channel, image2_with_expected_channel, image3_with_expected_channel],
            )
            == _EmptyImageList()
        )

        # Test remove image by index
        assert image_list.remove_image_by_index(0) == ImageList.from_images([image2, image3]).change_channel(
            image_list.channel,
        )
        assert image_list.remove_image_by_index(1) == ImageList.from_images([image1, image3]).change_channel(
            image_list.channel,
        )
        assert image_list.remove_image_by_index(2) == ImageList.from_images([image1, image2]).change_channel(
            image_list.channel,
        )
        assert image_list.remove_image_by_index([0, 1]) == ImageList.from_images([image3]).change_channel(
            image_list.channel,
        )
        assert image_list.remove_image_by_index([0, 2]) == ImageList.from_images([image2]).change_channel(
            image_list.channel,
        )
        assert image_list.remove_image_by_index([1, 2]) == ImageList.from_images([image1]).change_channel(
            image_list.channel,
        )
        assert image_list.remove_image_by_index([0, 1, 2]) == _EmptyImageList()

        # Test remove image by index ignore invalid
        assert image_list._remove_image_by_index_ignore_invalid(0) == ImageList.from_images(
            [image2, image3],
        ).change_channel(
            image_list.channel,
        )
        assert image_list._remove_image_by_index_ignore_invalid(1) == ImageList.from_images(
            [image1, image3],
        ).change_channel(
            image_list.channel,
        )
        assert image_list._remove_image_by_index_ignore_invalid(2) == ImageList.from_images(
            [image1, image2],
        ).change_channel(
            image_list.channel,
        )
        assert image_list._remove_image_by_index_ignore_invalid([0, 1]) == ImageList.from_images(
            [image3],
        ).change_channel(
            image_list.channel,
        )
        assert image_list._remove_image_by_index_ignore_invalid([0, 2]) == ImageList.from_images(
            [image2],
        ).change_channel(
            image_list.channel,
        )
        assert image_list._remove_image_by_index_ignore_invalid([1, 2]) == ImageList.from_images(
            [image1],
        ).change_channel(
            image_list.channel,
        )
        assert image_list._remove_image_by_index_ignore_invalid([0, 1, 2]) == _EmptyImageList()

        # Test remove images with size
        for image in [image1, image2, image3]:
            w, h = image.width, image.height
            assert image_list.remove_images_with_size(w, h) == ImageList.from_images(
                [
                    im
                    for im in [image1_with_expected_channel, image2_with_expected_channel, image3_with_expected_channel]
                    if im.width != w and im.height != h
                ],
            )
        assert image_list.remove_images_with_size(12345, 67890) == image_list

        # Test remove duplicate images
        assert image_list.remove_duplicate_images() == ImageList.from_images(
            [image1_with_expected_channel]
            + ([image2_with_expected_channel] if image2_with_expected_channel != image1_with_expected_channel else [])
            + (
                [image3_with_expected_channel]
                if image3_with_expected_channel not in [image1_with_expected_channel, image2_with_expected_channel]
                else []
            ),
        )

        # Test shuffle images
        image_list_shuffled = image_list.shuffle_images()
        assert len(image_list_shuffled) == 3
        assert image_list_shuffled.get_image(0) in image_list
        assert image_list_shuffled.get_image(1) in image_list
        assert image_list_shuffled.get_image(2) in image_list

        # Final Check for out-of-place
        assert image_list == image_list_clone


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestFromFiles:

    @pytest.mark.parametrize(
        "resource_path",
        [
            images_all(),
            str(test_images_folder),
            *images_all(),
            [Path(im) for im in images_all()],
            test_images_folder,
            *[Path(im) for im in images_all()],
        ],
        ids=[
            "all-images",
            "images_folder",
            *images_all_ids(),
            "all-images-path",
            "images_folder-path",
            *[s + "-path" for s in images_all_ids()],
        ],
    )
    def test_from_files_creation(self, resource_path: str | Path, snapshot_png_image_list: SnapshotAssertion, device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))
        image_list_returned_filenames, filenames = ImageList.from_files(
            resolve_resource_path(resource_path),
            return_filenames=True,
        )
        assert image_list == snapshot_png_image_list
        assert image_list == image_list_returned_filenames
        assert len(image_list) == len(filenames)

    @pytest.mark.parametrize(
        "resource_path",
        [
            "\\images\\missing_directory\\",
            Path("\\images\\missing_directory\\"),
            ["\\images\\missing_file1.png", "\\images\\missing_file2.png"],
            [Path("\\images\\missing_file1.png"), Path("\\images\\missing_file2.png")],
            [*images_all(), "\\images\\missing_file2.png"],
            [*[Path(im) for im in images_all()], Path("\\images\\missing_file2.png")],
        ],
        ids=["dir-str", "dir-path", "list-str", "list-path", "list-str-last-missing", "list-path-last-missing"],
    )
    def test_should_raise_if_one_file_or_directory_not_found(self, resource_path: str | Path, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(FileNotFoundError):
            ImageList.from_files(resolve_resource_path(resource_path))


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestToImages:

    @pytest.mark.parametrize(
        "resource_path",
        [images_all(), [plane_png_path, plane_jpg_path] * 2],
        ids=["all-images", "planes"],
    )
    def test_should_return_images(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list_all = ImageList.from_files(resolve_resource_path(resource_path))
        image_list_select = ImageList.from_files(resolve_resource_path(resource_path[::2]))
        assert image_list_all.to_images(list(range(0, len(image_list_all), 2))) == image_list_select.to_images()

    @pytest.mark.parametrize(
        "resource_path",
        [images_all(), [plane_png_path, plane_jpg_path]],
        ids=["all-images", "planes"],
    )
    def test_from_files_creation(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))
        bracket_open = r"\["
        bracket_close = r"\]"
        with pytest.raises(
            IndexOutOfBoundsError,
            match=rf"There are no elements at indices {str(list(range(len(image_list), 2 + len(image_list)))).replace('[', bracket_open).replace(']', bracket_close)}.",
        ):
            image_list.to_images(list(range(2, 2 + len(image_list))))


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestToJpegFiles:

    @pytest.mark.parametrize(
        "resource_path",
        [images_all(), [plane_png_path, plane_jpg_path]],
        ids=["all-images", "planes"],
    )
    def test_should_raise_if_alpha_channel(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(IllegalFormatError, match="This format is illegal. Use one of the following formats: png"),
        ):
            image_list.to_jpeg_files(tmpdir)

    @pytest.mark.parametrize(
        "resource_path",
        [
            [
                grayscale_jpg_path,
                plane_jpg_path,
                grayscale_jpg_path,
                plane_jpg_path,
                white_square_jpg_path,
                white_square_jpg_path,
                plane_jpg_path,
            ],
            [plane_jpg_path, plane_jpg_path],
            [grayscale_jpg_path, grayscale_jpg_path],
        ],
        ids=["all-jpg-images", "jpg-planes", "jpg-grayscale"],
    )
    def test_should_raise_if_invalid_path(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))
        with pytest.raises(
            ValueError,
            match="The path specified is invalid. Please provide either the path to a directory, a list of paths with one path for each image, or a list of paths with one path per image size.",
        ):
            image_list.to_jpeg_files([])

    @pytest.mark.parametrize(
        "resource_path",
        [
            [
                grayscale_jpg_path,
                plane_jpg_path,
                grayscale_jpg_path,
                plane_jpg_path,
                white_square_jpg_path,
                white_square_jpg_path,
                plane_jpg_path,
            ],
            [plane_jpg_path, plane_jpg_path],
            [grayscale_jpg_path, grayscale_jpg_path],
        ],
        ids=["all-jpg-images", "jpg-planes", "jpg-grayscale"],
    )
    def test_should_save_images_in_directory(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))
        with tempfile.TemporaryDirectory() as tmpdir:
            image_list.to_jpeg_files(tmpdir)
            image_list_loaded = ImageList.from_files(tmpdir)
            assert len(image_list) == len(image_list_loaded)
            assert image_list.number_of_sizes == image_list_loaded.number_of_sizes
            assert isinstance(image_list_loaded, type(image_list))
            for index in range(len(image_list)):
                im_saved = image_list.get_image(index)
                im_loaded = image_list_loaded.get_image(index)
                assert im_saved.width == im_loaded.width
                assert im_saved.height == im_loaded.height
                assert im_saved.channel == im_loaded.channel

    @pytest.mark.parametrize(
        "resource_path",
        [
            [
                grayscale_jpg_path,
                plane_jpg_path,
                grayscale_jpg_path,
                plane_jpg_path,
                white_square_jpg_path,
                white_square_jpg_path,
                plane_jpg_path,
            ],
            [plane_jpg_path, plane_jpg_path],
            [grayscale_jpg_path, grayscale_jpg_path],
        ],
        ids=["all-jpg-images", "jpg-planes", "jpg-grayscale"],
    )
    def test_should_save_images_in_directories_for_different_sizes(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))

        with tempfile.TemporaryDirectory() as tmp_parent_dir:
            tmp_dirs = [tempfile.TemporaryDirectory(dir=tmp_parent_dir) for _ in range(image_list.number_of_sizes)]

            image_list.to_jpeg_files([tmp_dir.name for tmp_dir in tmp_dirs])
            image_list_loaded = ImageList.from_files(tmp_parent_dir)
            assert len(image_list) == len(image_list_loaded)
            assert image_list.number_of_sizes == image_list_loaded.number_of_sizes
            assert isinstance(image_list_loaded, type(image_list))
            assert set(image_list.widths) == set(image_list_loaded.widths)
            assert set(image_list.heights) == set(image_list_loaded.heights)
            assert image_list.channel == image_list_loaded.channel
            assert set(image_list.sizes) == set(image_list_loaded.sizes)

            for tmp_dir in tmp_dirs:
                tmp_dir.cleanup()

    @pytest.mark.parametrize(
        "resource_path",
        [
            [
                grayscale_jpg_path,
                plane_jpg_path,
                grayscale_jpg_path,
                plane_jpg_path,
                white_square_jpg_path,
                white_square_jpg_path,
                plane_jpg_path,
            ],
            [plane_jpg_path, plane_jpg_path],
            [grayscale_jpg_path, grayscale_jpg_path],
        ],
        ids=["all-jpg-images", "jpg-planes", "jpg-grayscale"],
    )
    def test_should_save_images_in_files(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))

        with tempfile.TemporaryDirectory() as tmp_parent_dir:
            tmp_files = [
                tempfile.NamedTemporaryFile(suffix=".jpg", prefix=str(i), dir=tmp_parent_dir)
                for i in range(len(image_list))
            ]
            for tmp_file in tmp_files:
                tmp_file.close()

            image_list.to_jpeg_files([tmp_file.name for tmp_file in tmp_files])
            image_list_loaded = ImageList.from_files(tmp_parent_dir)
            assert len(image_list) == len(image_list_loaded)
            assert image_list.number_of_sizes == image_list_loaded.number_of_sizes
            assert isinstance(image_list_loaded, type(image_list))
            for index in range(len(image_list)):
                im_saved = image_list.get_image(index)
                im_loaded = image_list_loaded.get_image(index)
                assert im_saved.width == im_loaded.width
                assert im_saved.height == im_loaded.height
                assert im_saved.channel == im_loaded.channel


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestToPngFiles:

    @pytest.mark.parametrize(
        "resource_path",
        [images_all(), [plane_png_path, plane_jpg_path], [grayscale_png_path, grayscale_png_path]],
        ids=["all-images", "planes", "grayscale"],
    )
    def test_should_raise_if_invalid_path(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))
        with pytest.raises(
            ValueError,
            match="The path specified is invalid. Please provide either the path to a directory, a list of paths with one path for each image, or a list of paths with one path per image size.",
        ):
            image_list.to_png_files([])

    @pytest.mark.parametrize(
        "resource_path",
        [images_all(), [plane_png_path, plane_jpg_path], [grayscale_png_path, grayscale_png_path]],
        ids=["all-images", "planes", "grayscale"],
    )
    def test_should_save_images_in_directory(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))
        with tempfile.TemporaryDirectory() as tmpdir:
            image_list.to_png_files(tmpdir)
            image_list_loaded = ImageList.from_files(tmpdir)
            assert len(image_list) == len(image_list_loaded)
            assert image_list.number_of_sizes == image_list_loaded.number_of_sizes
            assert isinstance(image_list_loaded, type(image_list))
            assert image_list == image_list_loaded

    @pytest.mark.parametrize(
        "resource_path",
        [images_all(), [plane_png_path, plane_jpg_path], [grayscale_png_path, grayscale_png_path]],
        ids=["all-images", "planes", "grayscale"],
    )
    def test_should_save_images_in_directories_for_different_sizes(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))

        with tempfile.TemporaryDirectory() as tmp_parent_dir:
            tmp_dirs = [tempfile.TemporaryDirectory(dir=tmp_parent_dir) for _ in range(image_list.number_of_sizes)]

            image_list.to_png_files([tmp_dir.name for tmp_dir in tmp_dirs])
            image_list_loaded = ImageList.from_files(tmp_parent_dir)
            assert len(image_list) == len(image_list_loaded)
            assert image_list.number_of_sizes == image_list_loaded.number_of_sizes
            assert isinstance(image_list_loaded, type(image_list))
            assert set(image_list.widths) == set(image_list_loaded.widths)
            assert set(image_list.heights) == set(image_list_loaded.heights)
            assert image_list.channel == image_list_loaded.channel
            assert set(image_list.sizes) == set(image_list_loaded.sizes)

            for tmp_dir in tmp_dirs:
                tmp_dir.cleanup()

    @pytest.mark.parametrize(
        "resource_path",
        [images_all(), [plane_png_path, plane_jpg_path], [grayscale_png_path, grayscale_png_path]],
        ids=["all-images", "planes", "grayscale"],
    )
    def test_should_save_images_in_files(self, resource_path: list[str], device: Device) -> None:
        configure_test_with_device(device)
        image_list = ImageList.from_files(resolve_resource_path(resource_path))

        with tempfile.TemporaryDirectory() as tmp_parent_dir:
            tmp_files = [
                tempfile.NamedTemporaryFile(suffix=".png", prefix=str(i), dir=tmp_parent_dir)
                for i in range(len(image_list))
            ]
            for tmp_file in tmp_files:
                tmp_file.close()

            image_list.to_png_files([tmp_file.name for tmp_file in tmp_files])
            image_list_loaded = ImageList.from_files(tmp_parent_dir)
            assert len(image_list) == len(image_list_loaded)
            assert image_list.number_of_sizes == image_list_loaded.number_of_sizes
            assert isinstance(image_list_loaded, type(image_list))
            assert image_list == image_list_loaded


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestShuffleImages:

    @pytest.mark.parametrize(
        "resource_path",
        [images_all(), [plane_png_path, plane_jpg_path] * 2],
        ids=["all-images", "planes"],
    )
    def test_shuffle_images(self, resource_path: list[str], snapshot_png_image_list: SnapshotAssertion, device: Device) -> None:
        configure_test_with_device(device)
        image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
        image_list_clone = image_list_original._clone()
        random.seed(420)
        image_list_shuffled = image_list_original.shuffle_images()
        random.seed()
        assert len(image_list_shuffled) == len(resource_path)
        for index in range(len(resource_path)):
            assert image_list_shuffled.get_image(index) in image_list_original
        assert image_list_shuffled == snapshot_png_image_list
        assert image_list_original is not image_list_clone
        assert image_list_original == image_list_clone


@pytest.mark.parametrize("resource_path3", images_all_channel(), ids=images_all_channel_ids())
@pytest.mark.parametrize("resource_path2", images_all_channel(), ids=images_all_channel_ids())
@pytest.mark.parametrize("resource_path1", images_all_channel(), ids=images_all_channel_ids())
@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestTransformsEqualImageTransforms:

    @pytest.mark.parametrize(
        ("method", "attributes"),
        [
            ("resize", [2, 3]),
            ("resize", [50, 75]),
            ("resize", [700, 400]),
            ("convert_to_grayscale", None),
            ("crop", [0, 0, 100, 100]),
            ("flip_vertically", None),
            ("flip_horizontally", None),
            ("adjust_brightness", [0.5]),
            ("adjust_brightness", [10]),
            ("adjust_contrast", [0.75]),
            ("adjust_contrast", [5]),
            ("adjust_color_balance", [2]),
            ("adjust_color_balance", [0.5]),
            ("adjust_color_balance", [0]),
            ("blur", [2]),
            ("sharpen", [0]),
            ("sharpen", [0.5]),
            ("sharpen", [10]),
            ("invert_colors", None),
            ("rotate_right", None),
            ("rotate_left", None),
            ("find_edges", None),
        ],
        ids=[
            "resize-(2, 3)",
            "resize-(50, 75)",
            "resize-(700, 400)",
            "grayscale",
            "crop-(0, 0, 100, 100)",
            "flip_vertically",
            "flip_horizontally",
            "adjust_brightness-small factor",
            "adjust_brightness-large factor",
            "adjust_contrast-small factor",
            "adjust_contrast-large factor",
            "adjust_color_balance-add color",
            "adjust_color_balance-remove color",
            "adjust_color_balance-gray",
            "blur",
            "sharpen-zero factor",
            "sharpen-small factor",
            "sharpen-large factor",
            "invert_colors",
            "rotate_right",
            "rotate_left",
            "find_edges",
        ],
    )
    def test_all_transform_methods(
        self,
        method: str,
        attributes: list,
        resource_path1: str,
        resource_path2: str,
        resource_path3: str,
        device: Device,
    ) -> None:
        configure_test_with_device(device)
        image_list_original = ImageList.from_files(
            [
                resolve_resource_path(resource_path1),
                resolve_resource_path(resource_path2),
                resolve_resource_path(resource_path3),
            ],
        )
        image_list_clone = image_list_original._clone()

        if isinstance(attributes, list):
            image_list_transformed = getattr(image_list_original, method)(*attributes)
        else:
            image_list_transformed = getattr(image_list_original, method)()

        assert len(image_list_original) == len(image_list_transformed)
        assert image_list_original.channel == image_list_transformed.channel
        for index in range(len(image_list_original)):
            image_original = image_list_original.get_image(index)
            if isinstance(attributes, list):
                image_transformed = getattr(image_original, method)(*attributes)
            else:
                image_transformed = getattr(image_original, method)()
            assert image_transformed == image_list_transformed.get_image(index)
        assert image_list_original is not image_list_clone
        assert image_list_original == image_list_clone


@pytest.mark.parametrize(
    "resource_path",
    [images_all(), [plane_png_path, plane_jpg_path] * 2],
    ids=["all-images", "planes"],
)
@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestTransforms:
    class TestAddNoise:
        @pytest.mark.parametrize(
            "standard_deviation",
            [
                0.0,
                0.7,
                2.5,
            ],
            ids=["minimum noise", "some noise", "very noisy"],
        )
        def test_should_add_noise(
            self,
            resource_path: list[str],
            standard_deviation: float,
            snapshot_png_image_list: SnapshotAssertion,
            device: Device,
        ) -> None:
            skip_if_os([os_mac])
            configure_test_with_device(device)
            torch.manual_seed(0)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            image_list_noise = image_list_original.add_noise(standard_deviation)
            assert image_list_noise == snapshot_png_image_list
            assert image_list_original is not image_list_clone
            assert image_list_original == image_list_clone


@pytest.mark.parametrize(
    "resource_path",
    [images_all(), [plane_png_path, plane_jpg_path] * 2],
    ids=["SingleSizeImageList", "MultiSizeImageList"],
)
@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestErrorsAndWarningsWithoutEmptyImageList:

    class TestAddImageTensor:

        def test_should_raise(self, resource_path: list[str], device: Device) -> None:
            configure_test_with_device(device)
            image_list = ImageList.from_files(resolve_resource_path(resource_path))
            with pytest.raises(DuplicateIndexError, match=r"The index '0' is already in use."):
                image_list._add_image_tensor(image_list.to_images([0])[0]._image_tensor, 0)

    class TestEquals:

        def test_should_raise(self, resource_path: list[str], device: Device) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            assert (image_list_original.__eq__(image_list_original.to_images([0]))) is NotImplemented

    class TestCrop:

        @pytest.mark.parametrize(
            ("new_x", "new_y"),
            [(10000, 1), (1, 10000), (10000, 10000)],
            ids=["outside x", "outside y", "outside x and y"],
        )
        def test_should_warn_if_coordinates_outsize_image(
            self,
            resource_path: list[str],
            new_x: int,
            new_y: int,
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            image_list = ImageList.from_files(resolve_resource_path(resource_path))
            image_blank_tensor = torch.zeros((image_list.number_of_images, image_list.channel, 1, 1))
            with pytest.warns(
                UserWarning,
                match=r"The specified bounding rectangle does not contain any content of at least one image. Therefore these images will be blank.",
            ):
                cropped_image_list = image_list.crop(new_x, new_y, 1, 1)
                assert torch.all(torch.eq(cropped_image_list._as_single_size_image_list()._tensor, image_blank_tensor))

    class TestAdjustColorBalance:

        def test_should_not_adjust_color_balance_channel_1(
            self,
            resource_path: list[str],
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path)).change_channel(1)
            image_list_clone = image_list_original._clone()
            with pytest.warns(
                UserWarning,
                match="Color adjustment will not have an affect on grayscale images with only one channel.",
            ):
                image_list_no_change = image_list_original.adjust_color_balance(0.5)
                assert image_list_no_change is not image_list_original
                assert image_list_no_change == image_list_original
            assert image_list_original is not image_list_clone
            assert image_list_original == image_list_clone


@pytest.mark.parametrize(
    "resource_path",
    [images_all(), [plane_png_path, plane_jpg_path] * 2, []],
    ids=["SingleSizeImageList", "MultiSizeImageList", "EmptyImageList"],
)
@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestErrorsAndWarningsWithEmptyImageList:

    class TestChangeChannel:

        @pytest.mark.parametrize(
            "channel",
            [-1, 0, 2, 5],
            ids=["channel-negative-1", "channel-0", "channel-2", "channel-5"],
        )
        def test_should_raise(self, resource_path: list[str], channel: int, device: Device) -> None:
            configure_test_with_device(device)
            image_list = ImageList.from_files(resolve_resource_path(resource_path))
            with pytest.raises(
                ValueError,
                match=rf"Channel {channel} is not a valid channel option. Use either 1, 3 or 4",
            ):
                image_list.change_channel(channel)

    class TestRemoveImageByIndex:

        def test_should_raise_invalid_index(self, resource_path: list[str], device: Device) -> None:
            configure_test_with_device(device)
            image_list = ImageList.from_files(resolve_resource_path(resource_path))
            with pytest.raises(IndexOutOfBoundsError):
                image_list.remove_image_by_index(-1)
            with pytest.raises(IndexOutOfBoundsError):
                image_list.remove_image_by_index(len(image_list))

    class TestRemoveImagesWithSize:

        @pytest.mark.parametrize(
            ("width", "height"),
            [(-10, 10), (10, -10), (-10, -10)],
            ids=["invalid width", "invalid height", "invalid width and height"],
        )
        def test_should_raise_negative_size(self, resource_path: list[str], width: int, height: int, device: Device) -> None:
            configure_test_with_device(device)
            image_list = ImageList.from_files(resolve_resource_path(resource_path))
            with pytest.raises(
                OutOfBoundsError,
                match=rf"At least one of width and height \(={min(width, height)}\) is not inside \[1, \u221e\).",
            ):
                image_list.remove_images_with_size(width, height)

    class TestResize:

        @pytest.mark.parametrize(
            ("new_width", "new_height"),
            [(-10, 10), (10, -10), (-10, -10)],
            ids=["invalid width", "invalid height", "invalid width and height"],
        )
        def test_should_raise_new_size(self, resource_path: list[str], new_width: int, new_height: int, device: Device) -> None:
            configure_test_with_device(device)
            image_list = ImageList.from_files(resolve_resource_path(resource_path))
            with pytest.raises(
                OutOfBoundsError,
                match=rf"At least one of the new sizes new_width and new_height \(={min(new_width, new_height)}\) is not inside \[1, \u221e\).",
            ):
                image_list.resize(new_width, new_height)

    class TestCrop:

        @pytest.mark.parametrize(
            ("new_width", "new_height"),
            [(-10, 1), (1, -10), (-10, -1)],
            ids=["invalid width", "invalid height", "invalid width and height"],
        )
        def test_should_raise_invalid_size(self, resource_path: list[str], new_width: int, new_height: int, device: Device) -> None:
            configure_test_with_device(device)
            image_list = ImageList.from_files(resolve_resource_path(resource_path))
            with pytest.raises(
                OutOfBoundsError,
                match=rf"At least one of width and height \(={min(new_width, new_height)}\) is not inside \[1, \u221e\).",
            ):
                image_list.crop(0, 0, new_width, new_height)

        @pytest.mark.parametrize(
            ("new_x", "new_y"),
            [(-10, 1), (1, -10), (-10, -1)],
            ids=["invalid x", "invalid y", "invalid x and y"],
        )
        def test_should_raise_invalid_coordinates(self, resource_path: list[str], new_x: int, new_y: int, device: Device) -> None:
            configure_test_with_device(device)
            image_list = ImageList.from_files(resolve_resource_path(resource_path))
            with pytest.raises(
                OutOfBoundsError,
                match=rf"At least one of the coordinates x and y \(={min(new_x, new_y)}\) is not inside \[0, \u221e\).",
            ):
                image_list.crop(new_x, new_y, 100, 100)

    class TestAddNoise:

        @pytest.mark.parametrize(
            "standard_deviation",
            [-1],
            ids=["sigma below zero"],
        )
        def test_should_raise_standard_deviation(
            self,
            resource_path: list[str],
            standard_deviation: float,
            device: Device
        ) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.raises(
                OutOfBoundsError,
                match=rf"standard_deviation \(={standard_deviation}\) is not inside \[0, \u221e\)\.",
            ):
                image_list_original.add_noise(standard_deviation)
            assert image_list_original == image_list_clone

    class TestAdjustBrightness:

        @pytest.mark.parametrize(
            "factor",
            [-1],
            ids=["factor below zero"],
        )
        def test_should_raise(
            self,
            resource_path: list[str],
            factor: float,
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
                image_list_original.adjust_brightness(factor)
            assert image_list_original == image_list_clone

        def test_should_not_brighten(
            self,
            resource_path: list[str],
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.warns(
                UserWarning,
                match="Brightness adjustment factor is 1.0, this will not make changes to the images.",
            ):
                image_list_no_change = image_list_original.adjust_brightness(1)
                assert image_list_no_change == image_list_original
            assert image_list_original == image_list_clone

    class TestAdjustContrast:

        @pytest.mark.parametrize(
            "factor",
            [-1],
            ids=["factor below zero"],
        )
        def test_should_raise(
            self,
            resource_path: list[str],
            factor: float,
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
                image_list_original.adjust_contrast(factor)
            assert image_list_original == image_list_clone

        def test_should_not_adjust(
            self,
            resource_path: list[str],
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.warns(
                UserWarning,
                match="Contrast adjustment factor is 1.0, this will not make changes to the images.",
            ):
                image_list_no_change = image_list_original.adjust_contrast(1)
                assert image_list_no_change == image_list_original
            assert image_list_original == image_list_clone

    class TestAdjustColorBalance:

        @pytest.mark.parametrize(
            "factor",
            [-1],
            ids=["factor below zero"],
        )
        def test_should_raise(
            self,
            resource_path: list[str],
            factor: float,
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
                image_list_original.adjust_color_balance(factor)
            assert image_list_original == image_list_clone

        def test_should_not_adjust_color_balance_factor_1(
            self,
            resource_path: list[str],
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.warns(
                UserWarning,
                match="Color adjustment factor is 1.0, this will not make changes to the images.",
            ):
                image_list_no_change = image_list_original.adjust_color_balance(1)
                assert image_list_no_change == image_list_original
            assert image_list_original == image_list_clone

    class TestBlur:

        def test_should_raise_radius_out_of_bounds(self, resource_path: str, device: Device) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.raises(
                OutOfBoundsError,
                match=rf"radius \(=-1\) is not inside \[0, {'0' if isinstance(image_list_original, _EmptyImageList) else min(*image_list_original.widths, *image_list_original.heights) - 1}\].",
            ):
                image_list_original.blur(-1)
            with pytest.raises(
                OutOfBoundsError,
                match=rf"radius \(={'1' if isinstance(image_list_original, _EmptyImageList) else min(*image_list_original.widths, *image_list_original.heights)}\) is not inside \[0, {'0' if isinstance(image_list_original, _EmptyImageList) else min(*image_list_original.widths, *image_list_original.heights) - 1}\].",
            ):
                image_list_original.blur(
                    (
                        1
                        if isinstance(image_list_original, _EmptyImageList)
                        else min(*image_list_original.widths, *image_list_original.heights)
                    ),
                )
            assert image_list_original == image_list_clone

        def test_should_not_blur(self, resource_path: str, device: Device) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.warns(
                UserWarning,
                match="Blur radius is 0, this will not make changes to the images.",
            ):
                image_list_no_change = image_list_original.blur(0)
                assert image_list_no_change == image_list_original
            assert image_list_original == image_list_clone

    class TestSharpen:

        @pytest.mark.parametrize(
            "factor",
            [-1],
            ids=["factor below zero"],
        )
        def test_should_raise(
            self,
            resource_path: list[str],
            factor: float,
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
                image_list_original.sharpen(factor)
            assert image_list_original == image_list_clone

        def test_should_not_adjust(
            self,
            resource_path: list[str],
            device: Device,
        ) -> None:
            configure_test_with_device(device)
            image_list_original = ImageList.from_files(resolve_resource_path(resource_path))
            image_list_clone = image_list_original._clone()
            with pytest.warns(
                UserWarning,
                match="Sharpen factor is 1.0, this will not make changes to the images.",
            ):
                image_list_no_change = image_list_original.sharpen(1)
                assert image_list_no_change == image_list_original
            assert image_list_original == image_list_clone


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestSingleSizeImageList:

    @pytest.mark.parametrize(
        "tensor",
        [
            torch.ones(4, 1, 1),
        ],
    )
    def test_create_from_tensor_3_dim(self, tensor: Tensor, device: Device) -> None:
        configure_test_with_device(device)
        tensor = tensor.to(device)
        expected_tensor = tensor.unsqueeze(dim=1)
        image_list = _SingleSizeImageList._create_from_tensor(tensor, list(range(tensor.size(0))))
        assert image_list._tensor_positions_to_indices == list(range(tensor.size(0)))
        assert len(image_list) == expected_tensor.size(0)
        assert image_list.widths[0] == expected_tensor.size(3)
        assert image_list.heights[0] == expected_tensor.size(2)
        assert image_list.channel == expected_tensor.size(1)

    @pytest.mark.parametrize(
        "tensor",
        [
            torch.ones(4, 3, 1, 1),
        ],
    )
    def test_create_from_tensor_4_dim(self, tensor: Tensor, device: Device) -> None:
        configure_test_with_device(device)
        tensor = tensor.to(device)
        image_list = _SingleSizeImageList._create_from_tensor(tensor, list(range(tensor.size(0))))
        assert image_list._tensor_positions_to_indices == list(range(tensor.size(0)))
        assert len(image_list) == tensor.size(0)
        assert image_list.widths[0] == tensor.size(3)
        assert image_list.heights[0] == tensor.size(2)
        assert image_list.channel == tensor.size(1)

    @pytest.mark.parametrize("tensor", [torch.ones(4, 3, 1, 1, 1), torch.ones(4, 3)], ids=["5-dim", "2-dim"])
    def test_should_raise_from_invalid_tensor(self, tensor: Tensor, device: Device) -> None:
        configure_test_with_device(device)
        tensor = tensor.to(device)
        with pytest.raises(
            ValueError,
            match=rf"Invalid Tensor. This Tensor requires 3 or 4 dimensions but has {tensor.dim()}",
        ):
            _SingleSizeImageList._create_from_tensor(tensor, list(range(tensor.size(0))))

    @pytest.mark.parametrize(
        "tensor",
        [
            torch.randn(16, 4, 4),
        ],
    )
    def test_get_batch_and_iterate_3_dim(self, tensor: Tensor, device: Device) -> None:
        configure_test_with_device(device)
        tensor = tensor.to(device)
        expected_tensor = tensor.unsqueeze(dim=1)
        image_list = _SingleSizeImageList._create_from_tensor(tensor, list(range(tensor.size(0))))
        batch_size = math.ceil(expected_tensor.size(0) / 1.999)
        assert image_list._get_batch(0, batch_size).size(0) == batch_size
        assert torch.all(torch.eq(image_list._get_batch(0, 1), image_list._get_batch(0)))
        assert torch.all(
            torch.eq(image_list._get_batch(0, batch_size), expected_tensor[:batch_size].to(torch.float32) / 255),
        )
        assert torch.all(
            torch.eq(image_list._get_batch(1, batch_size), expected_tensor[batch_size:].to(torch.float32) / 255),
        )
        iterate_image_list = iter(image_list)
        assert iterate_image_list == image_list
        assert iterate_image_list is not image_list
        iterate_image_list._batch_size = batch_size
        assert torch.all(torch.eq(image_list._get_batch(0, batch_size), next(iterate_image_list)))
        assert torch.all(torch.eq(image_list._get_batch(1, batch_size), next(iterate_image_list)))
        with pytest.raises(IndexOutOfBoundsError, match=rf"There is no element at index '{batch_size * 2}'."):
            image_list._get_batch(2, batch_size)
        with pytest.raises(StopIteration):
            next(iterate_image_list)

    @pytest.mark.parametrize(
        "tensor",
        [
            torch.randn(16, 4, 4, 4),
        ],
    )
    def test_get_batch_and_iterate_4_dim(self, tensor: Tensor, device: Device) -> None:
        configure_test_with_device(device)
        tensor = tensor.to(device)
        image_list = _SingleSizeImageList._create_from_tensor(tensor, list(range(tensor.size(0))))
        batch_size = math.ceil(tensor.size(0) / 1.999)
        assert image_list._get_batch(0, batch_size).size(0) == batch_size
        assert torch.all(torch.eq(image_list._get_batch(0, 1), image_list._get_batch(0)))
        assert torch.all(torch.eq(image_list._get_batch(0, batch_size), tensor[:batch_size].to(torch.float32) / 255))
        assert torch.all(torch.eq(image_list._get_batch(1, batch_size), tensor[batch_size:].to(torch.float32) / 255))
        iterate_image_list = iter(image_list)
        assert iterate_image_list == image_list
        assert iterate_image_list is not image_list
        iterate_image_list._batch_size = batch_size
        assert torch.all(torch.eq(image_list._get_batch(0, batch_size), next(iterate_image_list)))
        assert torch.all(torch.eq(image_list._get_batch(1, batch_size), next(iterate_image_list)))
        with pytest.raises(IndexOutOfBoundsError, match=rf"There is no element at index '{batch_size * 2}'."):
            image_list._get_batch(2, batch_size)
        with pytest.raises(StopIteration):
            next(iterate_image_list)


@pytest.mark.parametrize("device", get_devices(), ids=get_devices_ids())
class TestEmptyImageList:

    def test_warn_empty_image_list(self, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.warns(
            UserWarning,
            match=r"You are using an empty ImageList. This method changes nothing if used on an empty ImageList.",
        ):
            _EmptyImageList._warn_empty_image_list()

    def test_create_image_list_in_empty_image_list(self, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(NotImplementedError):
            _EmptyImageList._create_image_list([], [])

    @pytest.mark.parametrize(
        "image_list",
        [_SingleSizeImageList(), _MultiSizeImageList()],
        ids=["SingleSizeImageList", "MultiSizeImageList"],
    )
    def test_create_image_list(self, image_list: ImageList, device: Device) -> None:
        configure_test_with_device(device)
        assert isinstance(image_list._create_image_list([], []), _EmptyImageList)

    def test_from_images(self, device: Device) -> None:
        configure_test_with_device(device)
        assert ImageList.from_images([]) == _EmptyImageList()

    def test_from_files(self, device: Device) -> None:
        configure_test_with_device(device)
        assert ImageList.from_files([]) == _EmptyImageList()
        with tempfile.TemporaryDirectory() as tmpdir:
            assert ImageList.from_files(tmpdir) == _EmptyImageList()
            assert ImageList.from_files([tmpdir]) == _EmptyImageList()

    def test_clone(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList() == _EmptyImageList()._clone()
        assert _EmptyImageList() is _EmptyImageList()._clone()  # Singleton

    def test_repr_png(self, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(TypeError, match=r"You cannot display an empty ImageList"):
            ImageList.from_images([])._repr_png_()

    def test_eq(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList() == _EmptyImageList()
        assert _EmptyImageList().__eq__(Table()) is NotImplemented

    def test_hash(self, device: Device) -> None:
        configure_test_with_device(device)
        assert hash(_EmptyImageList()) == hash(_EmptyImageList())

    def test_sizeof(self, device: Device) -> None:
        configure_test_with_device(device)
        assert sys.getsizeof(_EmptyImageList()) >= 0
        assert _EmptyImageList().__sizeof__() == 0

    def test_number_of_images(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList().number_of_images == 0

    def test_widths(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList().widths == []

    def test_heights(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList().heights == []

    def test_channel(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList().channel is NotImplemented

    def test_sizes(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList().sizes == []

    def test_number_of_sizes(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList().number_of_sizes == 0

    def test_get_image(self, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(IndexOutOfBoundsError, match=r"There is no element at index '0'."):
            _EmptyImageList().get_image(0)

    def test_index(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList().index(Image.from_file(resolve_resource_path(plane_png_path))) == []

    def test_has_image(self, device: Device) -> None:
        configure_test_with_device(device)
        assert not _EmptyImageList().has_image(Image.from_file(resolve_resource_path(plane_png_path)))
        assert Image.from_file(resolve_resource_path(plane_png_path)) not in _EmptyImageList()

    def test_to_jpeg_file(self, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.warns(UserWarning, match="You are using an empty ImageList. No files will be saved."):
            _EmptyImageList().to_jpeg_files("path")

    def test_to_png_file(self, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.warns(UserWarning, match="You are using an empty ImageList. No files will be saved."):
            _EmptyImageList().to_png_files("path")

    def test_to_images(self, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList().to_images() == []
        assert _EmptyImageList().to_images([0]) == []

    @pytest.mark.parametrize("resource_path", images_all(), ids=images_all_ids())
    def test_add_image_tensor(self, resource_path: str, device: Device) -> None:
        configure_test_with_device(device)
        assert _EmptyImageList()._add_image_tensor(
            Image.from_file(resolve_resource_path(resource_path))._image_tensor,
            0,
        ) == ImageList.from_files(resolve_resource_path(resource_path))

    def test_remove_image_by_index(self, device: Device) -> None:
        configure_test_with_device(device)
        with pytest.raises(IndexOutOfBoundsError):
            _EmptyImageList().remove_image_by_index(0)

    @pytest.mark.parametrize(
        ("method", "attributes"),
        [
            ("change_channel", [1]),
            ("_remove_image_by_index_ignore_invalid", [0]),
            ("_remove_image_by_index_ignore_invalid", [[0]]),
            ("remove_images_with_size", [2, 3]),
            ("remove_duplicate_images", None),
            ("shuffle_images", None),
            ("resize", [2, 3]),
            ("resize", [50, 75]),
            ("resize", [700, 400]),
            ("convert_to_grayscale", None),
            ("crop", [0, 0, 100, 100]),
            ("flip_vertically", None),
            ("flip_horizontally", None),
            ("adjust_brightness", [0.5]),
            ("adjust_brightness", [10]),
            ("add_noise", [10]),
            ("adjust_contrast", [0.75]),
            ("adjust_contrast", [5]),
            ("adjust_color_balance", [2]),
            ("adjust_color_balance", [0.5]),
            ("adjust_color_balance", [0]),
            ("blur", [0]),
            ("sharpen", [0]),
            ("sharpen", [0.5]),
            ("sharpen", [10]),
            ("invert_colors", None),
            ("rotate_right", None),
            ("rotate_left", None),
            ("find_edges", None),
        ],
        ids=[
            "change_channel-(1)",
            "_remove_image_by_index_ignore_invalid-(0)",
            "_remove_image_by_index_ignore_invalid-([0])",
            "remove_images_with_size-(2, 3)",
            "remove_duplicate_images",
            "shuffle_images",
            "resize-(2, 3)",
            "resize-(50, 75)",
            "resize-(700, 400)",
            "grayscale",
            "crop-(0, 0, 100, 100)",
            "flip_vertically",
            "flip_horizontally",
            "adjust_brightness-small factor",
            "adjust_brightness-large factor",
            "add_noise",
            "adjust_contrast-small factor",
            "adjust_contrast-large factor",
            "adjust_color_balance-add color",
            "adjust_color_balance-remove color",
            "adjust_color_balance-gray",
            "blur",
            "sharpen-zero factor",
            "sharpen-small factor",
            "sharpen-large factor",
            "invert_colors",
            "rotate_right",
            "rotate_left",
            "find_edges",
        ],
    )
    def test_transform_is_still_empty_image_list(self, method: str, attributes: list, device: Device) -> None:
        configure_test_with_device(device)
        image_list = _EmptyImageList()

        with pytest.warns(
            UserWarning,
            match=r"You are using an empty ImageList. This method changes nothing if used on an empty ImageList.",
        ):
            _EmptyImageList._warn_empty_image_list()
            if isinstance(attributes, list):
                image_list_transformed = getattr(image_list, method)(*attributes)
            else:
                image_list_transformed = getattr(image_list, method)()

        assert image_list_transformed == _EmptyImageList()
