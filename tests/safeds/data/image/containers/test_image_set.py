import random

import pytest
import torch
from syrupy import SnapshotAssertion

from safeds.config import _get_device
from safeds.data.image.containers import ImageSet, Image, _FixedSizedImageSet, _VariousSizedImageSet
from safeds.exceptions import IndexOutOfBoundsError, OutOfBoundsError, DuplicateIndexError
from tests.helpers import images_all, images_all_ids, resolve_resource_path, images_all_channel, images_all_channel_ids, \
    plane_png_path, plane_jpg_path


class TestAllImageCombinations:

    @pytest.mark.parametrize("resource_path3", images_all(), ids=images_all_ids())
    @pytest.mark.parametrize("resource_path2", images_all(), ids=images_all_ids())
    @pytest.mark.parametrize("resource_path1", images_all(), ids=images_all_ids())
    def test_from_files(self, resource_path1: str, resource_path2: str, resource_path3: str) -> None:
        # Setup
        torch.set_default_device(_get_device())

        image_set = ImageSet.from_files([resolve_resource_path(resource_path1), resolve_resource_path(resource_path2),
                                         resolve_resource_path(resource_path3)])
        image1 = Image.from_file(resolve_resource_path(resource_path1))
        image2 = Image.from_file(resolve_resource_path(resource_path2))
        image3 = Image.from_file(resolve_resource_path(resource_path3))
        expected_channel = max(image1.channel, image2.channel, image3.channel)
        images_not_included = []
        for image_path in images_all():
            if image_path[:-4] not in (resource_path1[:-4], resource_path2[:-4], resource_path3[:-4]):
                images_not_included.append(
                    Image.from_file(resolve_resource_path(image_path)).change_channel(expected_channel))

        image1_with_expected_channel = image1.change_channel(expected_channel)
        image2_with_expected_channel = image2.change_channel(expected_channel)
        image3_with_expected_channel = image3.change_channel(expected_channel)

        # Check if factory method selected the right ImageSet
        if image1.width == image2.width == image3.width and image1.height == image2.height == image3.height:
            assert isinstance(image_set, _FixedSizedImageSet)
            assert isinstance(image_set._as_fixed_sized_image_set(), _FixedSizedImageSet)
            with pytest.raises(ValueError, match=r"The given image_set is not a VariousSizedImageSet"):
                image_set._as_various_sized_image_set()
        else:
            assert isinstance(image_set, _VariousSizedImageSet)
            assert isinstance(image_set._as_various_sized_image_set(), _VariousSizedImageSet)
            with pytest.raises(ValueError, match=r"The given image_set is not a FixedSizedImageSet"):
                image_set._as_fixed_sized_image_set()

            # Check if all children are FixedSizedImageSets and have the right channel count
            for fixed_image_set in image_set._image_set_dict.values():
                assert isinstance(fixed_image_set, _FixedSizedImageSet)
                assert fixed_image_set.channel == expected_channel

        # Check if all images have the right index
        assert 0 in image_set.index(image1_with_expected_channel)
        assert 1 in image_set.index(image2_with_expected_channel)
        assert 2 in image_set.index(image3_with_expected_channel)
        for image in images_not_included:
            assert not image_set.index(image)
        with pytest.raises(IndexOutOfBoundsError, match=r"There is no element at index '3'."):
            image_set.get_image(3)
        assert image_set.get_image(0) == image1_with_expected_channel
        assert image_set.get_image(1) == image2_with_expected_channel
        assert image_set.get_image(2) == image3_with_expected_channel

        # Test eq
        image_set_equal = ImageSet.from_files(
            [resolve_resource_path(resource_path1), resolve_resource_path(resource_path2),
             resolve_resource_path(resource_path3)])
        image_set_unequal = ImageSet.from_images(images_not_included)
        assert image_set == image_set_equal
        assert image_set != image_set_unequal

        # Test from_images
        image_set_from_images = ImageSet.from_images([image1, image2, image3])
        assert image_set_from_images is not image_set
        assert image_set_from_images == image_set

        # Test clone
        image_set_clone = image_set.clone()
        assert image_set_clone is not image_set
        assert image_set_clone == image_set

        # Test len
        assert len(image_set) == 3

        # Test contains
        assert image1_with_expected_channel in image_set
        assert image2_with_expected_channel in image_set
        assert image3_with_expected_channel in image_set
        for image in images_not_included:
            assert image not in image_set

        # Test number_of_images
        assert image_set.number_of_images == 3

        # Test widths
        assert image_set.widths == [image1.width, image2.width, image3.width]

        # Test heights
        assert image_set.heights == [image1.height, image2.height, image3.height]

        # Test channel
        assert image_set.channel == expected_channel

        # Test number_of_sizes
        assert image_set.number_of_sizes == len(set((image.width, image.height) for image in [image1, image2, image3]))

        # Test has_image
        assert image_set.has_image(image1_with_expected_channel)
        assert image_set.has_image(image2_with_expected_channel)
        assert image_set.has_image(image3_with_expected_channel)
        for image in images_not_included:
            assert not image_set.has_image(image)

        # Test to_images
        assert image_set.to_images() == [image1_with_expected_channel, image2_with_expected_channel,
                                         image3_with_expected_channel]

        # Test change_channel
        assert image_set.change_channel(1).channel == 1
        assert image_set.change_channel(3).channel == 3
        assert image_set.change_channel(4).channel == 4

        # Test add image
        assert image_set == ImageSet.from_images([image1]).add_image(image2).add_image(image3)

        # Test add images
        assert image_set == ImageSet.from_images([image1]).add_images([image2, image3])
        assert image_set == ImageSet.from_images([image1]).add_images(ImageSet.from_images([image2, image3]))
        assert ImageSet.from_images([image1, image2, image3] + images_not_included) == image_set.add_images(
            images_not_included)

        # Test shuffle images
        image_set_shuffled = image_set.shuffle_images()
        assert len(image_set_shuffled) == 3
        assert image_set_shuffled.get_image(0) in image_set
        assert image_set_shuffled.get_image(1) in image_set
        assert image_set_shuffled.get_image(2) in image_set


class TestFromFiles:

    @pytest.mark.parametrize("resource_path", [images_all()], ids=["all-images"])
    def test_from_files_creation(self, resource_path: list[str], snapshot_png_image_set: SnapshotAssertion):
        torch.set_default_device(torch.device("cpu"))
        image_set = ImageSet.from_files(resolve_resource_path(resource_path))
        assert image_set == snapshot_png_image_set


class TestToImages:

    @pytest.mark.parametrize("resource_path", [images_all(), [plane_png_path, plane_jpg_path] * 2],
                             ids=["all-images", "planes"])
    def test_should_return_images(self, resource_path: list[str]):
        torch.set_default_device(torch.device("cpu"))
        image_set_all = ImageSet.from_files(resolve_resource_path(resource_path))
        image_set_select = ImageSet.from_files(resolve_resource_path(resource_path[::2]))
        assert image_set_all.to_images(list(range(0, len(image_set_all), 2))) == image_set_select.to_images()

    @pytest.mark.parametrize("resource_path", [images_all(), [plane_png_path, plane_jpg_path]],
                             ids=["all-images", "planes"])
    def test_from_files_creation(self, resource_path: list[str]):
        torch.set_default_device(torch.device("cpu"))
        image_set = ImageSet.from_files(resolve_resource_path(resource_path))
        bracket_open = r"\["
        bracket_close = r"\]"
        with pytest.raises(IndexOutOfBoundsError,
                           match=rf"There are no elements at indices {str(list(range(len(image_set), 2 + len(image_set)))).replace('[', bracket_open).replace(']', bracket_close)}."):
            image_set.to_images(list(range(2, 2 + len(image_set))))


class TestShuffleImages:

    @pytest.mark.parametrize("resource_path", [images_all(), [plane_png_path, plane_jpg_path] * 2],
                             ids=["all-images", "planes"])
    def test_shuffle_images(self, resource_path: list[str], snapshot_png_image_set: SnapshotAssertion):
        torch.set_default_device(_get_device())
        image_set_original = ImageSet.from_files(resolve_resource_path(resource_path))
        image_set_clone = image_set_original.clone()
        random.seed(420)
        image_set_shuffled = image_set_original.shuffle_images()
        random.seed()
        assert len(image_set_shuffled) == len(resource_path)
        for index in range(len(resource_path)):
            assert image_set_shuffled.get_image(index) in image_set_original
        assert image_set_shuffled == snapshot_png_image_set
        assert image_set_original is not image_set_clone
        assert image_set_original == image_set_clone


@pytest.mark.parametrize("resource_path3", images_all_channel(), ids=images_all_channel_ids())
@pytest.mark.parametrize("resource_path2", images_all_channel(), ids=images_all_channel_ids())
@pytest.mark.parametrize("resource_path1", images_all_channel(), ids=images_all_channel_ids())
class TestTransformsEqualImageTransforms:

    @pytest.mark.parametrize(
        ("method", "attributes"),
        [
            ('resize', [2, 3]),
            ('resize', [50, 75]),
            ('resize', [700, 400]),
            ('convert_to_grayscale', None),
            ('crop', [0, 0, 100, 100]),
            ('flip_vertically', None),
            ('flip_horizontally', None),
            ('adjust_brightness', [0.5]),
            ('adjust_brightness', [10]),
            ('adjust_contrast', [0.75]),
            ('adjust_contrast', [5]),
            ('adjust_color_balance', [2]),
            ('adjust_color_balance', [0.5]),
            ('adjust_color_balance', [0]),
            ('blur', [2]),
            ('sharpen', [0]),
            ('sharpen', [0.5]),
            ('sharpen', [10]),
            ('invert_colors', None),
            ('rotate_right', None),
            ('rotate_left', None),
            ('find_edges', None),
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
        ]
    )
    def test_all_transform_methods(self, method: str, attributes: list, resource_path1: str, resource_path2: str,
                                   resource_path3: str):
        torch.set_default_device(torch.device("cpu"))
        image_set_original = ImageSet.from_files(
            [resolve_resource_path(resource_path1), resolve_resource_path(resource_path2),
             resolve_resource_path(resource_path3)])
        image_set_clone = image_set_original.clone()

        if isinstance(attributes, list):
            image_set_transformed = getattr(image_set_original, method)(*attributes)
        else:
            image_set_transformed = getattr(image_set_original, method)()

        assert len(image_set_original) == len(image_set_transformed)
        assert image_set_original.channel == image_set_transformed.channel
        for index in range(len(image_set_original)):
            image_original = image_set_original.get_image(index)
            if isinstance(attributes, list):
                image_transformed = getattr(image_original, method)(*attributes)
            else:
                image_transformed = getattr(image_original, method)()
            assert image_transformed == image_set_transformed.get_image(index)
        assert image_set_original is not image_set_clone
        assert image_set_original == image_set_clone


@pytest.mark.parametrize("resource_path", [images_all(), [plane_png_path, plane_jpg_path] * 2],
                         ids=["all-images", "planes"])
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
            snapshot_png_image_set: SnapshotAssertion,
        ) -> None:
            torch.set_default_device(torch.device("cpu"))
            torch.manual_seed(0)
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            image_set_clone = image_set_original.clone()
            image_set_noise = image_set_original.add_noise(standard_deviation)
            assert image_set_noise == snapshot_png_image_set
            assert image_set_original is not image_set_clone
            assert image_set_original == image_set_clone


@pytest.mark.parametrize("resource_path", [images_all(), [plane_png_path, plane_jpg_path] * 2],
                         ids=["VariousSizedImageSet", "FixedSizedImageSet"])
class TestErrorsAndWarnings:

    class TestAddImageTensor:

        def test_should_raise(self, resource_path: list[str]):
            torch.set_default_device(torch.device("cpu"))
            image_set = ImageSet.from_files(resolve_resource_path(resource_path))
            with pytest.raises(DuplicateIndexError, match=r"The index '0' is already in use."):
                image_set._add_image_tensor(image_set.to_images([0])[0]._image_tensor, 0)

    class TestEquals:

        def test_should_raise(self, resource_path: list[str]):
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            assert (image_set_original.__eq__(image_set_original.to_images([0]))) is NotImplemented

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
        ) -> None:
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            image_set_clone = image_set_original.clone()
            with pytest.raises(
                OutOfBoundsError,
                match=rf"standard_deviation \(={standard_deviation}\) is not inside \[0, \u221e\)\.",
            ):
                image_set_original.add_noise(standard_deviation)
            assert image_set_original is not image_set_clone
            assert image_set_original == image_set_clone

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
        ) -> None:
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            image_set_clone = image_set_original.clone()
            with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
                image_set_original.adjust_brightness(factor)
            assert image_set_original is not image_set_clone
            assert image_set_original == image_set_clone

        def test_should_not_brighten(
            self,
            resource_path: list[str],
        ) -> None:
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            image_set_clone = image_set_original.clone()
            with pytest.warns(UserWarning,
                              match="Brightness adjustment factor is 1.0, this will not make changes to the images.", ):
                image_set_no_change = image_set_original.adjust_brightness(1)
                assert image_set_no_change is not image_set_original
                assert image_set_no_change == image_set_original
            assert image_set_original is not image_set_clone
            assert image_set_original == image_set_clone

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
        ) -> None:
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            image_set_clone = image_set_original.clone()
            with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
                image_set_original.adjust_contrast(factor)
            assert image_set_original is not image_set_clone
            assert image_set_original == image_set_clone

        def test_should_not_adjust(
            self,
            resource_path: list[str],
        ) -> None:
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            image_set_clone = image_set_original.clone()
            with pytest.warns(UserWarning,
                              match="Contrast adjustment factor is 1.0, this will not make changes to the images.", ):
                image_set_no_change = image_set_original.adjust_contrast(1)
                assert image_set_no_change is not image_set_original
                assert image_set_no_change == image_set_original
            assert image_set_original is not image_set_clone
            assert image_set_original == image_set_clone

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
        ) -> None:
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            image_set_clone = image_set_original.clone()
            with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
                image_set_original.adjust_color_balance(factor)
            assert image_set_original is not image_set_clone
            assert image_set_original == image_set_clone

        def test_should_not_adjust(
            self,
            resource_path: list[str],
        ) -> None:
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            image_set_clone = image_set_original.clone()
            with pytest.warns(UserWarning,
                              match="Color adjustment factor is 1.0, this will not make changes to the images.", ):
                image_set_no_change = image_set_original.adjust_color_balance(1)
                assert image_set_no_change is not image_set_original
                assert image_set_no_change == image_set_original
            assert image_set_original is not image_set_clone
            assert image_set_original == image_set_clone

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
        ) -> None:
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            image_set_clone = image_set_original.clone()
            with pytest.raises(OutOfBoundsError, match=r"factor \(=-1\) is not inside \[0, \u221e\)."):
                image_set_original.sharpen(factor)
            assert image_set_original is not image_set_clone
            assert image_set_original == image_set_clone

        def test_should_not_adjust(
            self,
            resource_path: list[str],
        ) -> None:
            image_set_original = ImageSet.from_files(
                [resolve_resource_path(unresolved_path) for unresolved_path in resource_path])
            image_set_clone = image_set_original.clone()
            with pytest.warns(UserWarning,
                              match="Sharpen factor is 1.0, this will not make changes to the images.", ):
                image_set_no_change = image_set_original.sharpen(1)
                assert image_set_no_change is not image_set_original
                assert image_set_no_change == image_set_original
            assert image_set_original is not image_set_clone
            assert image_set_original == image_set_clone
