from __future__ import annotations

from abc import ABCMeta, abstractmethod

import torch

from safeds.data.image.containers import Image
from torchvision.transforms.v2 import functional as func2


class ImageSet(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def create_image_set(images: list[Image]) -> ImageSet:
        pass

    @staticmethod
    def from_images(images: list[Image]) -> ImageSet:
        first_width = images[0].width
        first_height = images[0].height
        for image in images:
            if first_width != image.width or first_height != image.height:
                return _VariousSizedImageSet.create_image_set(images)
        return _FixedSizedImageSet.create_image_set(images)

    @abstractmethod
    def to_image_list(self) -> list[Image]:
        pass

    @abstractmethod
    def convert_to_grayscale(self) -> ImageSet:
        pass

    @abstractmethod
    def add_image(self, image: Image) -> ImageSet:
        pass

    def ready_for_nn(self) -> bool:
        return isinstance(self, _FixedSizedImageSet)


class _VariousSizedImageSet(ImageSet):

    def __init__(self):
        self._image_set_dict: dict[tuple[int, int], ImageSet] = {}

    def __len__(self):
        length = 0
        for image_set in self._image_set_dict.values():
            length += len(image_set)
        return length

    @staticmethod
    def create_image_set(images: list[Image]) -> ImageSet:
        image_set = _VariousSizedImageSet()

        for image in images:
            if (image.width, image.height) not in image_set._image_set_dict:
                image_set._image_set_dict[(image.width, image.height)] = _FixedSizedImageSet.create_image_set([image])
            else:
                image_set._image_set_dict[(image.width, image.height)] = image_set._image_set_dict.get(
                    (image.width, image.height)).add_image(image)

        return image_set

    def add_image(self, image: Image) -> ImageSet:
        image_set = _VariousSizedImageSet()
        image_set._image_set_dict = self._image_set_dict
        if (image.width, image.height) not in self._image_set_dict:
            image_set._image_set_dict[(image.width, image.height)] = _FixedSizedImageSet.create_image_set([image])
        else:
            image_set._image_set_dict[(image.width, image.height)] = self._image_set_dict.get(
                (image.width, image.height)).add_image(image)
        return image_set

    def convert_to_grayscale(self) -> ImageSet:
        image_set = _VariousSizedImageSet()
        for image_set_key, image_set_original in self._image_set_dict.items():
            image_set._image_set_dict[image_set_key] = image_set_original.convert_to_grayscale()
        return image_set

    def to_image_list(self) -> list[Image]:
        image_list = []
        for image_set in self._image_set_dict.values():
            for image in image_set.to_image_list():
                image_list.append(image)
        return image_list


class _FixedSizedImageSet(ImageSet):

    def __init__(self):
        self._tensor = None

    def __len__(self):
        return self._tensor.size(dim=0)

    @staticmethod
    def create_image_set(images: list[Image]) -> ImageSet:
        image_set = _FixedSizedImageSet()
        image_set._tensor = images[0]._image_tensor.unsqueeze(dim=0)
        for image in images[1:]:
            print(image_set._tensor.size())
            print(image._image_tensor.size())
            print(image._image_tensor.unsqueeze(dim=0).size())
            image_set._tensor = torch.cat([image_set._tensor, image._image_tensor.unsqueeze(dim=0)])
        return image_set

    def add_image(self, image: Image) -> ImageSet:
        if self._tensor.size(dim=2) == image.height and self._tensor.size(dim=3) == image.width:
            image_set = _FixedSizedImageSet()
            image_set._tensor = torch.cat([self._tensor, image._image_tensor.unsqueeze(dim=0)])
        else:
            image_set = _VariousSizedImageSet.create_image_set(self.to_image_list())
            image_set.add_image(image)
        return image_set

    def convert_to_grayscale(self) -> ImageSet:
        image_set = _FixedSizedImageSet()
        print(self._tensor[:, 0:3].size())
        image_set._tensor = func2.rgb_to_grayscale(self._tensor[:, 0:3])
        return image_set

    def to_image_list(self) -> list[Image]:
        image_list = []
        for i in range(self._tensor.size(dim=0)):
            image_list.append(Image(self._tensor[i]))
        return image_list
