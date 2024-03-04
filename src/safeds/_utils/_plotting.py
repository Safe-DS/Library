import io

import matplotlib.pyplot as plt

from safeds.data.image.containers import Image


def _create_image_for_plot(fig: plt.Figure) -> Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close()  # Prevents the figure from being displayed directly
    buffer.seek(0)
    return Image.from_bytes(buffer.read())
