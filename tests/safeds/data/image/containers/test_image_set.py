import pytest
import torch
from syrupy import SnapshotAssertion

from safeds.data.image.containers import ImageSet, Image, _FixedSizedImageSet, _VariousSizedImageSet
from safeds.exceptions import IndexOutOfBoundsError
from tests.helpers import images_all, images_all_ids, resolve_resource_path


class TestFromFiles:

    @pytest.mark.parametrize("resource_path3", images_all(), ids=images_all_ids())
    @pytest.mark.parametrize("resource_path2", images_all(), ids=images_all_ids())
    @pytest.mark.parametrize("resource_path1", images_all(), ids=images_all_ids())
    def test_from_files(self, resource_path1: str, resource_path2: str, resource_path3: str) -> None:
        image_set = ImageSet.from_files([resolve_resource_path(resource_path1), resolve_resource_path(resource_path2),
                                         resolve_resource_path(resource_path3)])
        image1 = Image.from_file(resolve_resource_path(resource_path1), torch.device("cpu"))
        image2 = Image.from_file(resolve_resource_path(resource_path2), torch.device("cpu"))
        image3 = Image.from_file(resolve_resource_path(resource_path3), torch.device("cpu"))
        images_not_included = []
        for image_path in images_all():
            if image_path[:-4] not in (resource_path1[:-4], resource_path2[:-4], resource_path3[:-4]):
                images_not_included.append(Image.from_file(resolve_resource_path(image_path), torch.device("cpu")))
        expected_channel = max(image1.channel, image2.channel, image3.channel)
        # Check if factory method selected the right ImageSet
        if image1.width == image2.width == image3.width and image1.height == image2.height == image3.height:
            assert isinstance(image_set, _FixedSizedImageSet)
            assert len(image_set.widths) == 1
            assert len(image_set.heights) == 1
            assert image_set.widths[0] == image1.width
            assert image_set.heights[0] == image1.height
        else:
            assert isinstance(image_set, _VariousSizedImageSet)
        # Check if the channel of the ImageSet is right
        assert image_set.channel == expected_channel
        # Check for the VariousSizedImageSets if all children are FixedSizedImageSets and have the right channel count
        if isinstance(image_set, _VariousSizedImageSet):
            for fixed_image_set in image_set._image_set_dict.values():
                assert isinstance(fixed_image_set, _FixedSizedImageSet)
                assert fixed_image_set.channel == expected_channel
        # Check if all images are in the ImageSet (with adjusted channels)
        assert image1.change_channel(expected_channel) in image_set
        assert image2.change_channel(expected_channel) in image_set
        assert image3.change_channel(expected_channel) in image_set
        for image in images_not_included:
            assert image.change_channel(expected_channel) not in image_set
        # Check if all images have the right index
        assert 0 in image_set.index(image1.change_channel(expected_channel))
        assert 1 in image_set.index(image2.change_channel(expected_channel))
        assert 2 in image_set.index(image3.change_channel(expected_channel))
        with pytest.raises(IndexOutOfBoundsError, match=r"There is no element at index '3'."):
            image_set.get_image(3)
        assert image_set.get_image(0) == image1.change_channel(expected_channel)
        assert image_set.get_image(1) == image2.change_channel(expected_channel)
        assert image_set.get_image(2) == image3.change_channel(expected_channel)

    @pytest.mark.parametrize("resource_path", [images_all()], ids=["all-images"])
    def test_from_files_creation(self, resource_path: list[str], snapshot_png_image_set: SnapshotAssertion):
        image_set = ImageSet.from_files(resolve_resource_path(resource_path))
        assert image_set == snapshot_png_image_set
