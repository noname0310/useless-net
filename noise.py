"""
create a noise image
"""
from typing import Tuple
import tensorflow as tf

def make_noise_image(image_size: Tuple[int, int] = (28, 28)) -> tf.Tensor:
    """
    Creates a noise image.
    Args:
        image_size: The size of the image.
    Returns:
        The noise image.
    """
    return tf.random.uniform(shape=(image_size[0], image_size[1], 3))
