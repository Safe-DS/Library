import torch

from safeds.config import _get_device


def test_device() -> None:
    if torch.cuda.is_available():
        assert _get_device() == torch.device('cuda')
    else:
        assert _get_device() == torch.device('cpu')
