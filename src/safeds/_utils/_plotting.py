from __future__ import annotations

import io
from typing import TYPE_CHECKING

from safeds.data.image.containers import Image

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _figure_to_image(figure: plt.Figure) -> Image:
    """
    Store the figure as an image and closes it.

    Parameters
    ----------
    figure:
        The figure to store.

    Returns
    -------
    image:
        The figure as an image.
    """
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)  # Prevents the figure from being displayed directly
    buffer.seek(0)
    return Image.from_bytes(buffer.read())
