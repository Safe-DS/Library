import torch

from safeds.data.image.containers import Image


class ImageSet:
    _image_dict = {}

    def __init__(self):
        pass

    def add_image(self, image: Image):
        self._image_dict[image.width, image.height] = image

    class _ImageSetTensor():

        def __init__(self):
